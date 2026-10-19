#!/usr/bin/env python3
"""Train and export a tiny integer-quantized vibration anomaly model.

The example models a high-rate, three-axis MPU6050 vibration stream.  Each
window is converted to one feature vector using the same operations a small
firmware FFT implementation can reproduce:

    vector magnitude -> mean removal -> Hann window -> one-sided FFT ->
    drop DC -> magnitude / (N / 2) -> log1p -> per-window z-score

The exported TFLite graph has a fixed ``[1, window_size / 2, 1]`` int8 input
and a fixed ``[1, 1]`` int8 output.  The output is the quantized form of a
sigmoid anomaly probability.  Input/output quantization parameters are
written to the metadata JSON and are also included as comments in the C++
header for firmware integration.

Only NumPy, TensorFlow, and Python's standard library are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Set this before importing TensorFlow so releases that read the environment
# during initialization can honor deterministic kernels as well.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# Import TensorFlow lazily enough that ``python -m py_compile`` and ``--help``
# still work on a machine that has NumPy but not TensorFlow.  The executable
# path reports a targeted error before doing any work when TensorFlow is absent.
_TF_IMPORT_ERROR: Optional[BaseException] = None
try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - depends on the local install
    tf = None  # type: ignore[assignment]
    _TF_IMPORT_ERROR = exc


_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_ROOT = _SCRIPT_PATH.parents[2]
_DEFAULT_ARTIFACTS_DIR = _PROJECT_ROOT / "predictive_maintenance" / "artifacts"
_DEFAULT_HEADER_PATH = (
    _PROJECT_ROOT / "predictive_maintenance" / "firmware" / "include" / "model.h"
)

_Z_SCORE_EPSILON = 1.0e-6
_DEFAULT_TFLITE_NAME = "vibration_anomaly_int8.tflite"
_DEFAULT_METADATA_NAME = "vibration_anomaly_metrics.json"

_BEARING_FEATURE_COLUMNS = (
    "max",
    "min",
    "mean",
    "sd",
    "rms",
    "skewness",
    "kurtosis",
    "crest",
    "form",
)
_BEARING_LABEL_COLUMN = "fault"
_EDGE_WINDOW_SIZE = 1024
_EDGE_SAMPLE_RATE_HZ = 1000.0
_EDGE_FEATURE_BINS = _EDGE_WINDOW_SIZE // 2
_RAW_AXIS_ALIASES = {
    "x": ("accel_x", "accelerometer_x", "axis_x", "x"),
    "y": ("accel_y", "accelerometer_y", "axis_y", "y"),
    "z": ("accel_z", "accelerometer_z", "axis_z", "z"),
}


def _require_tensorflow() -> Any:
    """Return TensorFlow or raise an actionable installation error."""

    if tf is None:
        detail = ""
        if _TF_IMPORT_ERROR is not None:
            detail = f" Import error: {_TF_IMPORT_ERROR}"
        raise RuntimeError(
            "TensorFlow is required to train and export this example, but it "
            f"could not be imported.{detail} Install a TensorFlow package and "
            "run the script again."
        )
    return tf


def _configure_determinism(seed: int) -> None:
    """Seed NumPy/TensorFlow and request deterministic TensorFlow kernels."""

    tensorflow = _require_tensorflow()
    # Newer TensorFlow releases expose the explicit API below; the environment
    # variable is already set before import for older releases.
    np.random.seed(seed)
    if hasattr(tensorflow.keras.utils, "set_random_seed"):
        tensorflow.keras.utils.set_random_seed(seed)
    else:  # Compatibility with older TensorFlow releases.
        tensorflow.random.set_seed(seed)
    enable_determinism = getattr(tensorflow.config.experimental, "enable_op_determinism", None)
    if enable_determinism is not None:
        enable_determinism()


def _validate_window_size(window_size: int) -> None:
    if window_size < 16:
        raise ValueError(
            f"window_size must be at least 16 samples; received {window_size}."
        )
    if window_size % 2 != 0:
        raise ValueError(
            f"window_size must be even so window_size/2 is an integer; received {window_size}."
        )


def _hann_window(window_size: int) -> np.ndarray:
    """Return the symmetric Hann window using its explicit device formula."""

    _validate_window_size(window_size)
    sample_index = np.arange(window_size, dtype=np.float64)
    # This is the symmetric Hann definition used by np.hanning.  Writing the
    # formula explicitly makes the MCU implementation unambiguous.
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * sample_index / (window_size - 1))
    return window.astype(np.float32)


def extract_fft_features(
    window: np.ndarray, hann_window: Optional[np.ndarray] = None
) -> np.ndarray:
    """Convert one ``[window_size, 3]`` vibration window to FFT features.

    The returned shape is ``[window_size / 2, 1]``.  The channel magnitude is
    computed before mean removal, matching the documented on-device pipeline.
    A near-constant spectrum is mapped to zeros rather than producing NaNs.
    """

    samples = np.asarray(window, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError(
            "Each vibration window must have shape [window_size, 3] for the "
            f"three MPU6050 axes; received {samples.shape}."
        )
    window_size = int(samples.shape[0])
    _validate_window_size(window_size)
    if not np.all(np.isfinite(samples)):
        raise ValueError("Vibration windows must contain only finite values.")

    if hann_window is None:
        hann = _hann_window(window_size)
    else:
        hann = np.asarray(hann_window, dtype=np.float32)
        if hann.shape != (window_size,):
            raise ValueError(
                f"hann_window must have shape [{window_size}], received {hann.shape}."
            )
        if not np.all(np.isfinite(hann)):
            raise ValueError("hann_window must contain only finite values.")

    # Use float64 for the scalar FFT preprocessing so the generated reference
    # data is stable across NumPy builds.  The model input is written as float32
    # immediately after normalization, just as firmware will provide it.
    magnitude = np.sqrt(
        np.sum(np.square(samples.astype(np.float64)), axis=1, dtype=np.float64)
    )
    mean_removed = magnitude - np.mean(magnitude, dtype=np.float64)
    windowed = mean_removed * hann.astype(np.float64)

    # rfft has window_size/2 + 1 bins for a real input.  Removing bin zero (DC)
    # therefore leaves exactly window_size/2 model features.
    # The explicit N/2 normalization makes this numerically compatible with
    # the small radix-2 FFT in firmware.  It also keeps the feature scale
    # stable if the window length is changed later.
    one_sided_magnitude = np.abs(np.fft.rfft(windowed))[1:] / (window_size / 2.0)
    logged = np.log1p(one_sided_magnitude)
    feature_mean = float(np.mean(logged, dtype=np.float64))
    feature_std = float(np.std(logged, dtype=np.float64))
    if not math.isfinite(feature_mean) or not math.isfinite(feature_std):
        raise ValueError("FFT preprocessing produced a non-finite feature vector.")

    if feature_std <= _Z_SCORE_EPSILON:
        normalized = np.zeros_like(logged, dtype=np.float64)
    else:
        normalized = (logged - feature_mean) / feature_std

    features = normalized.astype(np.float32).reshape(window_size // 2, 1)
    if not np.all(np.isfinite(features)):
        raise ValueError("FFT preprocessing produced non-finite model features.")
    return features


def _simulate_mpu6050_vibration_windows(
    count: int,
    window_size: int,
    sample_rate_hz: float,
    anomaly: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate deterministic, synthetic three-axis accelerometer windows."""

    if count < 1:
        raise ValueError(f"count must be positive; received {count}.")
    _validate_window_size(window_size)
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError(
            f"sample_rate_hz must be a finite positive number; received {sample_rate_hz}."
        )

    time = np.arange(window_size, dtype=np.float64) / float(sample_rate_hz)
    windows = np.empty((count, window_size, 3), dtype=np.float32)

    # These relative frequencies keep the simulated machine vibration below
    # Nyquist for every supported sample rate while still representing a
    # high-rate sensor stream (the default is 1 kHz, matching the firmware).
    base_relative_frequencies = np.asarray([0.045, 0.075, 0.110], dtype=np.float64)

    for window_index in range(count):
        relative_frequencies = base_relative_frequencies + rng.uniform(
            -0.004, 0.004, size=3
        )
        base_frequencies = sample_rate_hz * relative_frequencies
        phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
        amplitudes = rng.uniform(0.08, 0.22, size=3)

        signal = np.zeros((window_size, 3), dtype=np.float64)
        for axis in range(3):
            signal[:, axis] = amplitudes[axis] * np.sin(
                2.0 * np.pi * base_frequencies[axis] * time + phases[axis]
            )
            # A weak second harmonic makes the normal spectrum more realistic.
            signal[:, axis] += 0.025 * amplitudes[axis] * np.sin(
                2.0 * np.pi * base_frequencies[axis] * 2.01 * time
                + phases[axis] * 0.7
            )

        common_frequency = sample_rate_hz * rng.uniform(0.006, 0.018)
        common_motion = 0.04 * np.sin(
            2.0 * np.pi * common_frequency * time
            + rng.uniform(0.0, 2.0 * np.pi)
        )
        signal += common_motion[:, None] * rng.uniform(0.7, 1.2, size=3)[None, :]

        if anomaly:
            # A bearing/imbalance fault contributes a repeatable high-frequency
            # component plus a few short impacts.  Both are visible after the
            # magnitude and FFT preprocessing.
            # The MPU6050's accelerometer bandwidth is about 260 Hz with the
            # firmware's DLPF setting. Keep the simulated fault below that
            # limit while leaving it well above the normal machine harmonics.
            fault_frequency = sample_rate_hz * rng.uniform(0.180, 0.240)
            fault_phase = rng.uniform(0.0, 2.0 * np.pi)
            fault_amplitude = rng.uniform(0.18, 0.36)
            axis_weights = rng.uniform(0.65, 1.15, size=3)
            fault = np.sin(2.0 * np.pi * fault_frequency * time + fault_phase)
            fault_modulation = 1.0 + 0.25 * np.sin(
                2.0
                * np.pi
                * sample_rate_hz
                * rng.uniform(0.010, 0.035)
                * time
            )
            signal += (
                fault_amplitude
                * (fault * fault_modulation)[:, None]
                * axis_weights[None, :]
            )

            pulse_count = int(rng.integers(2, 5))
            sample_positions = rng.integers(0, window_size, size=pulse_count)
            for center in sample_positions:
                width = max(1, int(window_size * rng.uniform(0.010, 0.035)))
                distance = (np.arange(window_size, dtype=np.float64) - center) / width
                pulse = np.exp(-0.5 * np.square(distance))
                signal += pulse[:, None] * rng.uniform(0.12, 0.30, size=3)[None, :]

        # Model the static 1 g gravity vector on the z axis. The firmware reads
        # raw accelerometer axes and then forms their magnitude, so including
        # gravity here keeps the synthetic magnitude stream physically aligned
        # with a level-mounted sensor.
        signal[:, 2] += 1.0

        # Per-axis bias and sensor noise are deliberately present in both
        # classes. Mean removal in the feature pipeline prevents the bias from
        # becoming a shortcut for the classifier.
        signal += rng.normal(0.0, 0.045, size=(1, 3))
        noise_level = rng.uniform(0.012, 0.032)
        signal += rng.normal(0.0, noise_level, size=(window_size, 3))
        windows[window_index] = signal.astype(np.float32)

    return windows


def simulate_vibration_dataset(
    normal_windows: int,
    anomaly_windows: int,
    window_size: int,
    sample_rate_hz: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic raw windows and binary labels (0 normal, 1 anomaly)."""

    if normal_windows < 1 or anomaly_windows < 1:
        raise ValueError("normal_windows and anomaly_windows must both be positive.")
    rng = np.random.default_rng(seed)
    normal = _simulate_mpu6050_vibration_windows(
        normal_windows, window_size, sample_rate_hz, anomaly=False, rng=rng
    )
    anomalous = _simulate_mpu6050_vibration_windows(
        anomaly_windows, window_size, sample_rate_hz, anomaly=True, rng=rng
    )
    windows = np.concatenate((normal, anomalous), axis=0)
    labels = np.concatenate(
        (
            np.zeros(normal_windows, dtype=np.int32),
            np.ones(anomaly_windows, dtype=np.int32),
        ),
        axis=0,
    )
    order = rng.permutation(windows.shape[0])
    return windows[order], labels[order]


def _normalize_column_name(value: str) -> str:
    """Normalize a CSV header or label for schema comparisons."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_")


def _read_source_text(path: Path) -> Tuple[str, str]:
    """Read a CSV source, transparently selecting one member from a ZIP file."""

    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise ValueError(f"Data source does not exist: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Data source is not a file: {source_path}")

    is_zip = source_path.suffix.lower() == ".zip"
    if not is_zip:
        try:
            is_zip = zipfile.is_zipfile(source_path)
        except OSError as exc:
            raise ValueError(f"Could not inspect data source {source_path}: {exc}") from exc

    if is_zip:
        try:
            with zipfile.ZipFile(source_path) as archive:
                members = [
                    info.filename
                    for info in archive.infolist()
                    if not info.is_dir() and info.filename.lower().endswith(".csv")
                ]
                if not members:
                    raise ValueError(
                        f"ZIP data source {source_path} contains no CSV member."
                    )
                preferred = [
                    member
                    for member in members
                    if Path(member).name.lower() == "b30hz40.csv"
                ]
                if len(preferred) == 1:
                    member = preferred[0]
                elif len(members) == 1:
                    member = members[0]
                else:
                    raise ValueError(
                        f"ZIP data source {source_path} contains multiple CSV members; "
                        f"select one by supplying a ZIP with one CSV (members: {members})."
                    )
                try:
                    raw_bytes = archive.read(member)
                except KeyError as exc:  # pragma: no cover - defensive archive boundary
                    raise ValueError(
                        f"Could not read CSV member {member!r} from {source_path}."
                    ) from exc
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Data source is not a valid ZIP file: {source_path}") from exc
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"CSV member in {source_path} is not valid UTF-8 text."
            ) from exc
        return text, f"{source_path}!{member}"

    try:
        text = source_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV data source is not valid UTF-8 text: {source_path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read data source {source_path}: {exc}") from exc
    return text, str(source_path)


def _csv_dialect(text: str) -> csv.Dialect:
    """Detect a common delimited-text dialect without requiring pandas."""

    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        # The supplied files are comma-separated.  The fallback also gives a
        # useful header/row validation error for malformed files.
        return csv.excel


def _normalized_csv_header(text: str, source_name: str) -> Tuple[List[str], csv.Dialect]:
    if not text.strip():
        raise ValueError(f"CSV data source is empty: {source_name}")
    dialect = _csv_dialect(text)
    reader = csv.reader(io.StringIO(text), dialect=dialect)
    try:
        raw_header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"CSV data source has no header row: {source_name}") from exc
    header = [_normalize_column_name(value) for value in raw_header]
    if not header or any(not value for value in header):
        raise ValueError(f"CSV header contains an empty column name: {source_name}")
    duplicates = sorted(
        value for value, count in Counter(header).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"CSV header has duplicate columns after normalization {duplicates}: {source_name}"
        )
    return header, dialect


def _read_csv_header(path: Path) -> Tuple[List[str], str]:
    text, source_name = _read_source_text(path)
    header, _ = _normalized_csv_header(text, source_name)
    return header, source_name


def _read_csv_table(path: Path) -> Tuple[List[str], List[Dict[str, str]], str]:
    """Read a small, validated table using only Python's csv module."""

    text, source_name = _read_source_text(path)
    header, dialect = _normalized_csv_header(text, source_name)
    reader = csv.reader(io.StringIO(text), dialect=dialect)
    # The first row was consumed by _normalized_csv_header's independent reader.
    try:
        next(reader)
    except StopIteration as exc:  # pragma: no cover - guarded by header validation
        raise ValueError(f"CSV data source has no header row: {source_name}") from exc

    rows: List[Dict[str, str]] = []
    for row_number, values in enumerate(reader, start=2):
        if not values or all(not str(value).strip() for value in values):
            continue
        if len(values) != len(header):
            raise ValueError(
                f"CSV row {row_number} in {source_name} has {len(values)} fields; "
                f"expected {len(header)} from the header."
            )
        rows.append({column: value.strip() for column, value in zip(header, values)})
    if not rows:
        raise ValueError(f"CSV data source has no data rows: {source_name}")
    return header, rows, source_name


def _find_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    column_set = set(columns)
    for alias in aliases:
        normalized = _normalize_column_name(alias)
        if normalized in column_set:
            return normalized
    return None


def _schema_for_columns(columns: Sequence[str]) -> str:
    """Classify the supplied datasets from their normalized headers."""

    column_set = set(columns)
    if set(_BEARING_FEATURE_COLUMNS).issubset(column_set) and _BEARING_LABEL_COLUMN in column_set:
        return "bearing_features"
    if all(_find_column(columns, aliases) is not None for aliases in _RAW_AXIS_ALIASES.values()):
        return "raw_accelerometer"
    if {"a1", "a2", "a3"}.issubset(column_set):
        return "gearbox_accelerometer"
    return "unknown"


def detect_data_schema(path: Path) -> str:
    """Return the supported schema name for a CSV or CSV-containing ZIP."""

    columns, source_name = _read_csv_header(Path(path))
    schema = _schema_for_columns(columns)
    if schema == "unknown":
        raise ValueError(
            f"Could not detect a supported schema in {source_name}. Expected either "
            "accel_x/accel_y/accel_z, a1/a2/a3, or bearing feature columns plus fault; "
            f"found columns {columns}."
        )
    return schema


def _parse_finite_float(
    value: str, source_name: str, row_number: int, column: str
) -> float:
    if value is None or not str(value).strip():
        raise ValueError(
            f"Missing numeric value in {source_name}, row {row_number}, column {column!r}."
        )
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric value in {source_name}, row {row_number}, column {column!r}: "
            f"{value!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Non-finite numeric value in {source_name}, row {row_number}, "
            f"column {column!r}: {value!r}."
        )
    return parsed


def _validate_window_extraction_args(window_size: int, window_stride: Optional[int]) -> int:
    _validate_window_size(window_size)
    if window_size != _EDGE_WINDOW_SIZE:
        raise ValueError(
            "Real-data window extraction must use exactly 1024 samples for the "
            f"ESP32-S3 contract; received {window_size}."
        )
    stride = window_size if window_stride is None else int(window_stride)
    if stride < 1:
        raise ValueError(f"window_stride must be at least 1; received {stride}.")
    return stride


def _stream_windows_from_rows(
    rows: Sequence[Mapping[str, str]],
    source_name: str,
    columns: Sequence[str],
    used_columns: Sequence[str],
    window_size: int,
    window_stride: Optional[int],
    schema: str,
    source_annotation: str,
    ignored_columns: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    stride = _validate_window_extraction_args(window_size, window_stride)
    if not rows:
        raise ValueError(f"No samples found in {source_name}.")

    samples = np.empty((len(rows), len(used_columns)), dtype=np.float32)
    for row_index, row in enumerate(rows, start=2):
        for axis_index, column in enumerate(used_columns):
            samples[row_index - 2, axis_index] = _parse_finite_float(
                row.get(column), source_name, row_index, column
            )
        # The gearbox file carries a fourth channel.  It is not part of the
        # ESP32 [x, y, z] contract, but validating it prevents a silently
        # corrupted replay from being accepted as an anomaly source.
        for column in ignored_columns:
            if column.startswith("a") and column[1:].isdigit():
                _parse_finite_float(row.get(column), source_name, row_index, column)

    total_samples = int(samples.shape[0])
    if total_samples < window_size:
        raise ValueError(
            f"{source_name} contains {total_samples} samples, fewer than the requested "
            f"window_size {window_size}."
        )
    starts = list(range(0, total_samples - window_size + 1, stride))
    if not starts:
        raise ValueError(
            f"{source_name} does not contain one complete window of {window_size} samples."
        )
    windows = np.asarray(
        [samples[start : start + window_size] for start in starts], dtype=np.float32
    )
    last_consumed = starts[-1] + window_size
    return windows, {
        "source": source_name,
        "schema": schema,
        "axis_columns": list(used_columns),
        "ignored_columns": list(ignored_columns),
        "source_annotation": source_annotation,
        "source_has_labels": False,
        "ground_truth_label": False,
        "total_samples": total_samples,
        "window_size": int(window_size),
        "window_stride": int(stride),
        "complete_windows": int(windows.shape[0]),
        "dropped_samples": int(total_samples - last_consumed),
    }


def load_raw_vibration_windows(
    path: Path, window_size: int, window_stride: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a labelled-by-source normal accelerometer stream into FFT windows."""

    columns, rows, source_name = _read_csv_table(Path(path))
    schema = _schema_for_columns(columns)
    if schema != "raw_accelerometer":
        raise ValueError(
            f"Raw vibration source {source_name} has schema {schema!r}; expected "
            "accel_x/accel_y/accel_z columns."
        )
    used_columns = [
        _find_column(columns, _RAW_AXIS_ALIASES[axis]) for axis in ("x", "y", "z")
    ]
    if any(column is None for column in used_columns):  # pragma: no cover - schema guard
        raise ValueError(f"Could not resolve all three accelerometer axes in {source_name}.")
    resolved_columns = [str(column) for column in used_columns]
    ignored_columns = [column for column in columns if column not in resolved_columns]
    return _stream_windows_from_rows(
        rows,
        source_name,
        columns,
        resolved_columns,
        window_size,
        window_stride,
        schema,
        "Kirloskar source-labelled normal calibration/replay",
        ignored_columns,
    )


def load_gearbox_vibration_windows(
    path: Path, window_size: int, window_stride: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a1/a2/a3 gearbox replay windows as source-labelled anomalies."""

    columns, rows, source_name = _read_csv_table(Path(path))
    schema = _schema_for_columns(columns)
    if schema != "gearbox_accelerometer":
        raise ValueError(
            f"Gearbox source {source_name} has schema {schema!r}; expected a1/a2/a3 "
            "(and optionally a4) columns."
        )
    used_columns = ["a1", "a2", "a3"]
    ignored_columns = [column for column in columns if column not in used_columns]
    return _stream_windows_from_rows(
        rows,
        source_name,
        columns,
        used_columns,
        window_size,
        window_stride,
        schema,
        "gearbox source-labelled anomaly/raw replay",
        ignored_columns,
    )


def _bearing_class(label: str) -> str:
    return "normal" if _normalize_column_name(label).startswith("normal") else "anomaly"


def load_bearing_feature_rows(path: Path) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """Load bearing feature rows and calculate deterministic calibration metadata."""

    columns, rows, source_name = _read_csv_table(Path(path))
    schema = _schema_for_columns(columns)
    if schema != "bearing_features":
        raise ValueError(
            f"Bearing source {source_name} has schema {schema!r}; expected the nine "
            f"feature columns plus {_BEARING_LABEL_COLUMN!r}."
        )

    feature_values = np.empty((len(rows), len(_BEARING_FEATURE_COLUMNS)), dtype=np.float32)
    labels: List[str] = []
    for row_index, row in enumerate(rows, start=2):
        for feature_index, column in enumerate(_BEARING_FEATURE_COLUMNS):
            feature_values[row_index - 2, feature_index] = _parse_finite_float(
                row.get(column), source_name, row_index, column
            )
        label = str(row.get(_BEARING_LABEL_COLUMN, "")).strip()
        if not label:
            raise ValueError(
                f"Missing bearing label in {source_name}, row {row_index}, "
                f"column {_BEARING_LABEL_COLUMN!r}."
            )
        labels.append(label)

    label_counts = Counter(labels)
    class_counts = Counter(_bearing_class(label) for label in labels)
    calibration_statistics: Dict[str, Dict[str, Any]] = {}
    for feature_index, column in enumerate(_BEARING_FEATURE_COLUMNS):
        values = feature_values[:, feature_index].astype(np.float64)
        calibration_statistics[column] = {
            "count": int(values.size),
            "mean": float(np.mean(values, dtype=np.float64)),
            "std": float(np.std(values, dtype=np.float64)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    metadata: Dict[str, Any] = {
        "source": source_name,
        "schema": schema,
        "row_count": int(feature_values.shape[0]),
        "feature_columns": list(_BEARING_FEATURE_COLUMNS),
        "label_column": _BEARING_LABEL_COLUMN,
        "source_has_labels": True,
        "label_counts": dict(sorted(label_counts.items())),
        "class_counts": {
            "normal": int(class_counts.get("normal", 0)),
            "anomaly": int(class_counts.get("anomaly", 0)),
        },
        "class_rule": "labels whose normalized value starts with 'normal' are normal; all other labels are anomaly",
        "calibration_statistics": calibration_statistics,
        "used_for_fft_training": False,
        "training_note": (
            "Bearing rows are feature-level calibration/provenance data. They are not "
            "fed to the model because the ESP32 contract accepts FFT bins from raw [x, y, z] windows."
        ),
    }
    return feature_values, labels, metadata


def load_real_dataset(
    raw_path: Path,
    bearing_path: Path,
    gear_path: Path,
    window_size: int,
    window_stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load the three supplied sources for the deterministic real-data mode."""

    normal_windows, normal_metadata = load_raw_vibration_windows(
        raw_path, window_size=window_size, window_stride=window_stride
    )
    anomaly_windows, anomaly_metadata = load_gearbox_vibration_windows(
        gear_path, window_size=window_size, window_stride=window_stride
    )
    _, _, bearing_metadata = load_bearing_feature_rows(bearing_path)

    if normal_windows.shape[0] < 3 or anomaly_windows.shape[0] < 3:
        raise ValueError(
            "Real-data mode needs at least 3 complete windows from each source so the "
            "stratified train/validation/test split has every class in every partition; "
            f"received normal={normal_windows.shape[0]}, anomaly={anomaly_windows.shape[0]}."
        )

    windows = np.concatenate((normal_windows, anomaly_windows), axis=0)
    labels = np.concatenate(
        (
            np.zeros(normal_windows.shape[0], dtype=np.int32),
            np.ones(anomaly_windows.shape[0], dtype=np.int32),
        ),
        axis=0,
    )
    metadata: Dict[str, Any] = {
        "mode": "real",
        "sources": {
            "normal": normal_metadata,
            "anomaly": anomaly_metadata,
        },
        "bearing": bearing_metadata,
        "window_order": "all Kirloskar normal windows followed by all gearbox anomaly windows before the seeded split",
        "class_counts": {
            "normal": int(normal_windows.shape[0]),
            "anomaly": int(anomaly_windows.shape[0]),
        },
        "assumptions": [
            "Unlabeled raw streams are source-labelled normal/gearbox for bring-up, not ground truth.",
            "The unlabeled Kirloskar raw stream is source-labelled normal for calibration/replay bring-up, not ground truth.",
            "The unlabeled a1/a2/a3 gearbox raw stream is source-labelled anomaly for replay bring-up, not ground truth.",
            "The bearing fault labels and feature statistics are source metadata and calibration evidence; they are not labels for the FFT-window model.",
            "Raw windows use contiguous, non-overlapping windows by default and discard only an incomplete trailing remainder.",
            "The raw CSVs do not provide a sampling-rate contract; sample_rate_hz is recorded as the ESP32 preprocessing assumption and must be verified before deployment.",
        ],
    }
    return windows, labels, metadata


def _discover_real_data_paths(data_dir: Path) -> Dict[str, Path]:
    """Find exactly one source of each supported schema under a directory."""

    root = Path(data_dir).expanduser()
    if not root.exists():
        raise ValueError(f"Real-data directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Real-data path is not a directory: {root}")

    candidates: Dict[str, List[Path]] = {
        "raw": [],
        "bearing": [],
        "gear": [],
    }
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.name.startswith("."):
            continue
        if candidate.suffix.lower() not in {".csv", ".zip"}:
            continue
        try:
            schema = detect_data_schema(candidate)
        except (OSError, UnicodeError, ValueError, zipfile.BadZipFile):
            continue
        role = {
            "raw_accelerometer": "raw",
            "bearing_features": "bearing",
            "gearbox_accelerometer": "gear",
        }.get(schema)
        if role is not None:
            candidates[role].append(candidate)

    missing = [role for role, paths in candidates.items() if not paths]
    ambiguous = {
        role: [str(path) for path in paths]
        for role, paths in candidates.items()
        if len(paths) > 1
    }
    if missing or ambiguous:
        details: List[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if ambiguous:
            details.append(
                "multiple candidates for "
                + "; ".join(f"{role}: {paths}" for role, paths in ambiguous.items())
            )
        raise ValueError(
            f"Could not unambiguously discover real datasets in {root}: "
            + "; ".join(details)
            + ". Supply --raw-path, --bearing-path, and --gear-path explicitly."
        )
    return {role: paths[0] for role, paths in candidates.items()}


def _real_data_requested(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None)
        for name in ("real_data_dir", "raw_path", "bearing_path", "gear_path")
    )


def _resolve_real_data_paths(args: argparse.Namespace) -> Dict[str, Path]:
    root: Optional[Path] = None
    if args.real_data_dir:
        root = Path(args.real_data_dir).expanduser()
        if not root.exists():
            raise ValueError(f"Real-data directory does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Real-data path is not a directory: {root}")

    explicit: Dict[str, Optional[str]] = {
        "raw": args.raw_path,
        "bearing": args.bearing_path,
        "gear": args.gear_path,
    }
    resolved: Dict[str, Path] = {}
    for role, value in explicit.items():
        if not value:
            continue
        candidate = Path(value).expanduser()
        if root is not None and not candidate.is_absolute():
            root_candidate = root / candidate
            candidate = root_candidate if root_candidate.exists() else candidate
        resolved[role] = candidate

    if root is not None and len(resolved) < len(explicit):
        discovered = _discover_real_data_paths(root)
        for role, candidate in discovered.items():
            resolved.setdefault(role, candidate)

    missing = [role for role in ("raw", "bearing", "gear") if role not in resolved]
    if missing:
        raise ValueError(
            "Real-data mode requires all three sources (raw, bearing, gear); missing "
            + ", ".join(missing)
            + ". Provide --real-data-dir or the explicit --raw-path, --bearing-path, and --gear-path options."
        )

    expected_schemas = {
        "raw": "raw_accelerometer",
        "bearing": "bearing_features",
        "gear": "gearbox_accelerometer",
    }
    for role, candidate in resolved.items():
        if not candidate.exists():
            raise ValueError(f"{role} data source does not exist: {candidate}")
        actual_schema = detect_data_schema(candidate)
        if actual_schema != expected_schemas[role]:
            raise ValueError(
                f"{role} data source {candidate} has schema {actual_schema!r}; "
                f"expected {expected_schemas[role]!r}."
            )
        resolved[role] = candidate.resolve()
    return resolved


def _extract_dataset_features(windows: np.ndarray) -> np.ndarray:
    samples = np.asarray(windows, dtype=np.float32)
    if samples.ndim != 3 or samples.shape[2] != 3:
        raise ValueError(
            "The raw dataset must have shape [num_windows, window_size, 3]; "
            f"received {samples.shape}."
        )
    if samples.shape[0] < 1:
        raise ValueError("The raw dataset is empty.")
    hann = _hann_window(int(samples.shape[1]))
    return np.stack(
        [extract_fft_features(window, hann_window=hann) for window in samples], axis=0
    ).astype(np.float32)


def _stratified_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Split each class into train/validation/test sets using NumPy only."""

    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int32).reshape(-1)
    if feature_array.ndim != 3 or feature_array.shape[0] != label_array.shape[0]:
        raise ValueError("features and labels have incompatible shapes for splitting.")
    if not np.all(np.isin(label_array, (0, 1))):
        raise ValueError("labels must contain only binary values 0 (normal) and 1 (anomaly).")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    if test_fraction + validation_fraction >= 1.0:
        raise ValueError("test_fraction + validation_fraction must be less than 1.")

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    validation_indices: List[int] = []
    test_indices: List[int] = []

    for class_label in (0, 1):
        class_indices = np.flatnonzero(label_array == class_label).astype(np.int64)
        if class_indices.size < 3:
            raise ValueError(
                f"Need at least 3 windows for class {class_label} to make a "
                f"stratified split; received {class_indices.size}."
            )
        rng.shuffle(class_indices)
        test_count = max(1, int(round(class_indices.size * test_fraction)))
        validation_count = max(1, int(round(class_indices.size * validation_fraction)))
        # The minimum class size check above leaves at least one training item.
        while test_count + validation_count >= class_indices.size:
            if test_count > 1 and test_count >= validation_count:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                break
        if test_count + validation_count >= class_indices.size:
            raise ValueError(
                f"Class {class_label} has too few windows ({class_indices.size}) "
                "for the requested train/validation/test split."
            )
        test_indices.extend(class_indices[:test_count].tolist())
        validation_indices.extend(
            class_indices[test_count : test_count + validation_count].tolist()
        )
        train_indices.extend(class_indices[test_count + validation_count :].tolist())

    for index_group in (train_indices, validation_indices, test_indices):
        rng.shuffle(index_group)

    def select(index_group: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(index_group, dtype=np.int64)
        return feature_array[indices], label_array[indices].astype(np.float32).reshape(-1, 1)

    train_features, train_labels = select(train_indices)
    validation_features, validation_labels = select(validation_indices)
    test_features, test_labels = select(test_indices)
    return {
        "train_features": train_features,
        "train_labels": train_labels,
        "validation_features": validation_features,
        "validation_labels": validation_labels,
        "test_features": test_features,
        "test_labels": test_labels,
    }


def _build_model(feature_bins: int, learning_rate: float) -> Any:
    """Build a small 1D CNN with a sigmoid anomaly-probability output."""

    tensorflow = _require_tensorflow()
    if feature_bins < 8:
        raise ValueError(f"feature_bins must be at least 8; received {feature_bins}.")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be a finite positive number; received {learning_rate}."
        )

    inputs = tensorflow.keras.Input(
        shape=(feature_bins, 1), dtype=tensorflow.float32, name="fft_features"
    )
    x = tensorflow.keras.layers.Conv1D(
        filters=8, kernel_size=5, padding="same", activation="relu", name="conv_1"
    )(inputs)
    x = tensorflow.keras.layers.MaxPooling1D(pool_size=2, name="pool_1")(x)
    x = tensorflow.keras.layers.Conv1D(
        filters=12, kernel_size=3, padding="same", activation="relu", name="conv_2"
    )(x)
    x = tensorflow.keras.layers.GlobalAveragePooling1D(name="global_average")(x)
    x = tensorflow.keras.layers.Dense(8, activation="relu", name="dense_1")(x)
    outputs = tensorflow.keras.layers.Dense(
        1, activation="sigmoid", name="anomaly_probability"
    )(x)
    model = tensorflow.keras.Model(inputs=inputs, outputs=outputs, name="vibration_cnn")
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tensorflow.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def _fit_model(
    model: Any,
    split: Mapping[str, np.ndarray],
    epochs: int,
    batch_size: int,
    verbose: int,
) -> Dict[str, List[float]]:
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1; received {epochs}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1; received {batch_size}.")
    history = model.fit(
        split["train_features"],
        split["train_labels"],
        validation_data=(split["validation_features"], split["validation_labels"]),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=verbose,
    )
    return {
        str(key): [float(value) for value in values]
        for key, values in history.history.items()
    }


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    """Compute dependency-free binary metrics at a probability threshold of 0.5."""

    actual = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if actual.size == 0 or actual.size != scores.size:
        raise ValueError("labels and probabilities must be non-empty and equally sized.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model probabilities must be finite.")
    predicted = (scores >= 0.5).astype(np.int32)
    true_positive = int(np.sum((actual == 1) & (predicted == 1)))
    true_negative = int(np.sum((actual == 0) & (predicted == 0)))
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    specificity = ratio(true_negative, true_negative + false_positive)
    f1 = ratio(2.0 * precision * recall, precision + recall)
    return {
        "count": int(actual.size),
        "threshold": 0.5,
        "accuracy": float(np.mean(predicted == actual)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "brier_score": float(np.mean(np.square(scores - actual))),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "mean_probability_normal": float(np.mean(scores[actual == 0]))
        if np.any(actual == 0)
        else None,
        "mean_probability_anomaly": float(np.mean(scores[actual == 1]))
        if np.any(actual == 1)
        else None,
    }


def _representative_dataset(
    features: np.ndarray, maximum_samples: int
) -> Iterable[List[np.ndarray]]:
    """Yield one float32, batch-one sample at a time for TFLite calibration."""

    if maximum_samples < 1:
        raise ValueError(
            f"representative_samples must be at least 1; received {maximum_samples}."
        )
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 3 or feature_array.shape[0] == 0:
        raise ValueError("Representative features must be a non-empty [N, bins, 1] array.")
    sample_count = min(maximum_samples, feature_array.shape[0])
    selected_indices = np.linspace(
        0, feature_array.shape[0] - 1, num=sample_count, dtype=np.int64
    )
    for index in selected_indices:
        # The fixed batch dimension is intentional: it is the firmware input.
        yield [feature_array[int(index) : int(index) + 1].astype(np.float32, copy=False)]


def _convert_to_int8(
    model: Any, representative_features: np.ndarray, feature_bins: int, representative_samples: int
) -> bytes:
    """Convert a Keras model to a fixed-shape, full-integer TFLite model."""

    tensorflow = _require_tensorflow()

    @tensorflow.function(
        input_signature=[
            tensorflow.TensorSpec(
                shape=(1, feature_bins, 1),
                dtype=tensorflow.float32,
                name="fft_features",
            )
        ]
    )
    def serving_function(features: Any) -> Any:
        return model(features, training=False)

    try:
        concrete_function = serving_function.get_concrete_function()
        converter = tensorflow.lite.TFLiteConverter.from_concrete_functions(
            [concrete_function], model
        )
        converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: _representative_dataset(
            representative_features, representative_samples
        )
        converter.target_spec.supported_ops = [
            tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tensorflow.int8
        converter.inference_output_type = tensorflow.int8
        return bytes(converter.convert())
    except Exception as exc:  # pragma: no cover - exact failures depend on TF
        raise RuntimeError(
            "Full-integer TFLite conversion failed. Ensure the model uses only "
            f"TFLite-supported layers and representative data is valid: {exc}"
        ) from exc


def _read_quantization_parameters(detail: Mapping[str, Any], tensor_name: str) -> Dict[str, Any]:
    """Read and validate per-tensor int8 quantization parameters."""

    dtype = np.dtype(detail.get("dtype"))
    if dtype != np.dtype(np.int8):
        raise RuntimeError(
            f"TFLite {tensor_name} dtype must be int8 for full integer I/O; received {dtype}."
        )

    parameters = detail.get("quantization_parameters", {})
    scales = np.asarray(parameters.get("scales", []), dtype=np.float64).reshape(-1)
    zero_points = np.asarray(parameters.get("zero_points", []), dtype=np.int64).reshape(-1)
    if scales.size != 1 or zero_points.size != 1:
        # Older interpreters expose the same information through the tuple.
        legacy_scale, legacy_zero_point = detail.get("quantization", (0.0, 0))
        scales = np.asarray([legacy_scale], dtype=np.float64)
        zero_points = np.asarray([legacy_zero_point], dtype=np.int64)
    scale = float(scales[0])
    zero_point = int(zero_points[0])
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(
            f"TFLite {tensor_name} quantization scale must be finite and positive; "
            f"received {scale}."
        )
    if zero_point < -128 or zero_point > 127:
        raise RuntimeError(
            f"TFLite {tensor_name} zero point must fit int8; received {zero_point}."
        )
    return {
        "dtype": "int8",
        "scale": scale,
        "zero_point": zero_point,
        "quantization_formula": "real_value = (quantized_value - zero_point) * scale",
    }


def _validate_int8_tflite(
    model_content: bytes,
    features: np.ndarray,
    labels: np.ndarray,
    feature_bins: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run every test window through the int8 interpreter and validate its contract."""

    tensorflow = _require_tensorflow()
    try:
        interpreter = tensorflow.lite.Interpreter(model_content=model_content, num_threads=1)
        interpreter.allocate_tensors()
    except Exception as exc:  # pragma: no cover - exact failures depend on TF
        raise RuntimeError(f"Could not load the generated TFLite model: {exc}") from exc

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(
            "Expected exactly one TFLite input and one output; received "
            f"{len(input_details)} inputs and {len(output_details)} outputs."
        )
    input_detail = input_details[0]
    output_detail = output_details[0]
    input_shape = tuple(int(value) for value in input_detail["shape"])
    output_shape = tuple(int(value) for value in output_detail["shape"])
    expected_input_shape = (1, feature_bins, 1)
    if input_shape != expected_input_shape:
        raise RuntimeError(
            "TFLite input shape must be exactly [1, window_size/2, 1]; "
            f"received {list(input_shape)}, expected {list(expected_input_shape)}."
        )
    if output_shape != (1, 1):
        raise RuntimeError(
            f"TFLite output shape must be [1, 1] for a sigmoid probability; received {list(output_shape)}."
        )

    input_quantization = _read_quantization_parameters(input_detail, "input")
    output_quantization = _read_quantization_parameters(output_detail, "output")

    # Integer inference may contain int32 accumulators, but no floating-point
    # tensors.  This catches accidental partial quantization that still has a
    # float operator or constant in the graph.
    tensor_dtypes = sorted(
        {str(np.dtype(detail.get("dtype"))) for detail in interpreter.get_tensor_details()}
    )
    floating_dtypes = {"float16", "float32", "float64"}
    if floating_dtypes.intersection(tensor_dtypes):
        raise RuntimeError(
            "TFLite graph is not fully integer quantized; floating tensor dtypes "
            f"were found: {sorted(floating_dtypes.intersection(tensor_dtypes))}."
        )

    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels).reshape(-1)
    if feature_array.ndim != 3 or feature_array.shape[1:] != (feature_bins, 1):
        raise ValueError(
            f"TFLite validation features must have shape [N, {feature_bins}, 1]; "
            f"received {feature_array.shape}."
        )
    if feature_array.shape[0] != label_array.shape[0] or feature_array.shape[0] == 0:
        raise ValueError("TFLite validation features and labels must be non-empty and aligned.")

    input_index = int(input_detail["index"])
    output_index = int(output_detail["index"])
    input_scale = input_quantization["scale"]
    input_zero_point = input_quantization["zero_point"]
    output_scale = output_quantization["scale"]
    output_zero_point = output_quantization["zero_point"]
    int8_limits = np.iinfo(np.int8)
    probabilities: List[float] = []
    for sample_index, sample in enumerate(feature_array):
        quantized_input = np.rint(sample[None, ...] / input_scale + input_zero_point)
        quantized_input = np.clip(
            quantized_input, int8_limits.min, int8_limits.max
        ).astype(np.int8)
        try:
            interpreter.set_tensor(input_index, quantized_input)
            interpreter.invoke()
            quantized_output = interpreter.get_tensor(output_index)
        except Exception as exc:  # pragma: no cover - exact failures depend on TF
            raise RuntimeError(
                f"TFLite int8 invocation failed for validation sample {sample_index}: {exc}"
            ) from exc
        probability = (
            np.asarray(quantized_output, dtype=np.float32).reshape(-1)
            - output_zero_point
        ) * output_scale
        if probability.size != 1 or not np.isfinite(probability[0]):
            raise RuntimeError(
                f"TFLite output for validation sample {sample_index} was not one finite value."
            )
        probabilities.append(float(np.clip(probability[0], 0.0, 1.0)))

    return np.asarray(probabilities, dtype=np.float32), {
        "input": {
            **input_quantization,
            "shape": list(input_shape),
            "name": str(input_detail.get("name", "")),
        },
        "output": {
            **output_quantization,
            "shape": list(output_shape),
            "name": str(output_detail.get("name", "")),
        },
        "tensor_dtypes": tensor_dtypes,
    }


def _ensure_output_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create output directory {path.parent}: {exc}") from exc
    if path.exists() and path.is_dir():
        raise OSError(f"Output path points to a directory, not a file: {path}")


def _render_model_header(
    model_content: bytes,
    input_shape: Sequence[int],
    input_quantization: Mapping[str, Any],
    output_quantization: Mapping[str, Any],
) -> str:
    """Render a self-contained C++17-compatible byte-array header."""

    bytes_per_line = 12
    byte_lines: List[str] = []
    for start in range(0, len(model_content), bytes_per_line):
        chunk = model_content[start : start + bytes_per_line]
        suffix = "," if start + bytes_per_line < len(model_content) else ""
        byte_lines.append(
            "  " + ", ".join(f"0x{byte:02x}" for byte in chunk) + suffix
        )
    input_scale = float(input_quantization["scale"])
    input_zero_point = int(input_quantization["zero_point"])
    output_scale = float(output_quantization["scale"])
    output_zero_point = int(output_quantization["zero_point"])
    lines = [
        "// Auto-generated by predictive_maintenance/training/train_vibration_model.py.",
        "// Do not edit: regenerate this file when the model or preprocessing changes.",
        f"// TFLite input: int8 {list(input_shape)}; scale={input_scale:.9g}; zero_point={input_zero_point}",
        f"// TFLite output: int8 [1, 1]; scale={output_scale:.9g}; zero_point={output_zero_point}",
        "// Dequantize output probability as (q - zero_point) * scale.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        # The firmware sketch includes this header directly and consumes the
        # conventional xxd-style symbols g_model and g_model_len.
        "alignas(16) const unsigned char g_model[] = {",
    ]
    lines.extend(byte_lines)
    lines.extend(
        [
            "};",
            "",
            "const int g_model_len = static_cast<int>(sizeof(g_model));",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    tflite_path: Path,
    metadata_path: Path,
    header_path: Path,
    model_content: bytes,
    metadata: Mapping[str, Any],
    header_text: str,
) -> None:
    for output_path in (tflite_path, metadata_path, header_path):
        _ensure_output_parent(output_path)
    try:
        tflite_path.write_bytes(model_content)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        header_path.write_text(header_text, encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not write generated artifacts: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny CNN from deterministic synthetic vibration by default, or "
            "ingest the supplied raw/bearing/gearbox CSV sources and export a fully "
            "integer-quantized TFLite model."
        )
    )
    parser.add_argument("--seed", type=int, default=2025, help="Random seed (default: 2025).")
    parser.add_argument(
        "--window-size",
        type=int,
        default=1024,
        help="Samples per three-axis window; fixed at 1024 for ESP32-S3 (default: 1024).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=1000.0,
        help=(
            "Sample-rate assumption in Hz; fixed at 1000 for ESP32-S3 and recorded "
            "for real-data provenance (default: 1000)."
        ),
    )
    parser.add_argument(
        "--normal-windows",
        type=int,
        default=120,
        help="Number of normal windows to simulate (default: 120).",
    )
    parser.add_argument(
        "--anomaly-windows",
        type=int,
        default=120,
        help="Number of anomalous windows to simulate (default: 120).",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help=(
            "Stride for real-data raw window extraction; defaults to window-size "
            "for contiguous non-overlapping windows."
        ),
    )
    parser.add_argument(
        "--real-data-dir",
        default=None,
        help=(
            "Directory containing one raw accelerometer CSV, one bearing-feature "
            "CSV, and one gearbox CSV/ZIP; enables real-data mode with schema discovery."
        ),
    )
    parser.add_argument(
        "--raw-path",
        "--raw-data-path",
        "--raw-data",
        "--raw-csv",
        dest="raw_path",
        default=None,
        help="Explicit normal raw accelerometer CSV path for real-data mode.",
    )
    parser.add_argument(
        "--bearing-path",
        "--bearing-data-path",
        "--bearing-data",
        "--bearing-csv",
        dest="bearing_path",
        default=None,
        help="Explicit bearing feature CSV path for real-data mode.",
    )
    parser.add_argument(
        "--gear-path",
        "--gear-data-path",
        "--gear-data",
        "--gearbox-path",
        "--gearbox-data-path",
        "--gearbox-data",
        "--gearbox-csv",
        dest="gear_path",
        default=None,
        help="Explicit gearbox raw CSV or CSV ZIP path for real-data mode.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Per-class fraction held out for final testing (default: 0.20).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Per-class fraction held out for validation (default: 0.20).",
    )
    parser.add_argument(
        "--epochs", type=int, default=12, help="Training epochs (default: 12)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size (default: 16)."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
        help="Adam learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=100,
        help="Maximum training samples used for int8 calibration (default: 100).",
    )
    parser.add_argument(
        "--artifacts-dir",
        "--output-dir",
        dest="artifacts_dir",
        default=str(_DEFAULT_ARTIFACTS_DIR),
        help=(
            "Directory for the default TFLite and metadata files "
            f"(default: {_DEFAULT_ARTIFACTS_DIR})."
        ),
    )
    parser.add_argument(
        "--tflite-path",
        "--tflite-output",
        dest="tflite_path",
        default=None,
        help="Explicit .tflite output path; otherwise it is placed in artifacts-dir.",
    )
    parser.add_argument(
        "--metadata-path",
        "--metadata-output",
        dest="metadata_path",
        default=None,
        help="Explicit metrics/metadata .json path; otherwise it is placed in artifacts-dir.",
    )
    parser.add_argument(
        "--header-path",
        "--header-output",
        dest="header_path",
        default=str(_DEFAULT_HEADER_PATH),
        help=f"C++ model.h output path (default: {_DEFAULT_HEADER_PATH}).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Keras fit verbosity: 0, 1, or 2 (default: 2).",
    )
    return parser


def _resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    artifacts_dir = Path(args.artifacts_dir).expanduser()
    tflite_path = (
        Path(args.tflite_path).expanduser()
        if args.tflite_path
        else artifacts_dir / _DEFAULT_TFLITE_NAME
    )
    metadata_path = (
        Path(args.metadata_path).expanduser()
        if args.metadata_path
        else artifacts_dir / _DEFAULT_METADATA_NAME
    )
    header_path = Path(args.header_path).expanduser()
    if tflite_path.suffix.lower() != ".tflite":
        raise ValueError(f"tflite output must end in .tflite; received {tflite_path}.")
    if metadata_path.suffix.lower() != ".json":
        raise ValueError(f"metadata output must end in .json; received {metadata_path}.")
    normalized_paths = [path.expanduser().resolve() for path in (tflite_path, metadata_path, header_path)]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("tflite, metadata, and header outputs must be different files.")
    return tflite_path, metadata_path, header_path


def _validate_args(args: argparse.Namespace) -> None:
    _validate_window_size(args.window_size)
    if args.window_size != _EDGE_WINDOW_SIZE:
        raise ValueError(
            "window_size must be exactly 1024 for the ESP32-S3/dashboard contract; "
            f"received {args.window_size}."
        )
    if args.window_stride is not None and args.window_stride < 1:
        raise ValueError(
            f"window_stride must be at least 1; received {args.window_stride}."
        )
    if not math.isfinite(args.sample_rate_hz) or args.sample_rate_hz <= 0.0:
        raise ValueError(
            f"sample_rate_hz must be a finite positive number; received {args.sample_rate_hz}."
        )
    if not math.isclose(args.sample_rate_hz, _EDGE_SAMPLE_RATE_HZ, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            "sample_rate_hz must be exactly 1000 Hz for the ESP32-S3/dashboard "
            f"contract; received {args.sample_rate_hz}."
        )
    if not _real_data_requested(args) and (
        args.normal_windows < 3 or args.anomaly_windows < 3
    ):
        raise ValueError(
            "normal_windows and anomaly_windows must each be at least 3 so "
            "train/validation/test each contain both labels."
        )
    if not (0.0 < args.test_fraction < 1.0):
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    if not (0.0 < args.validation_fraction < 1.0):
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    if args.test_fraction + args.validation_fraction >= 1.0:
        raise ValueError("test_fraction + validation_fraction must be less than 1.")
    if args.epochs < 1:
        raise ValueError(f"epochs must be at least 1; received {args.epochs}.")
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be at least 1; received {args.batch_size}.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be a finite positive number; received {args.learning_rate}."
        )
    if args.representative_samples < 1:
        raise ValueError(
            "representative_samples must be at least 1; "
            f"received {args.representative_samples}."
        )
    if args.seed < 0:
        raise ValueError(f"seed must be non-negative; received {args.seed}.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        tflite_path, metadata_path, header_path = _resolve_output_paths(args)
        real_data_metadata: Optional[Dict[str, Any]] = None
        if _real_data_requested(args):
            real_paths = _resolve_real_data_paths(args)
            raw_windows, labels, real_data_metadata = load_real_dataset(
                raw_path=real_paths["raw"],
                bearing_path=real_paths["bearing"],
                gear_path=real_paths["gear"],
                window_size=args.window_size,
                window_stride=args.window_stride,
            )
            real_data_metadata["sample_rate_hz_assumption"] = float(args.sample_rate_hz)
            data_mode = "real"
        else:
            raw_windows, labels = simulate_vibration_dataset(
                normal_windows=args.normal_windows,
                anomaly_windows=args.anomaly_windows,
                window_size=args.window_size,
                sample_rate_hz=args.sample_rate_hz,
                seed=args.seed,
            )
            data_mode = "synthetic"

        # Keep ingestion and validation usable for a NumPy-only smoke check;
        # TensorFlow is needed only once the real or synthetic arrays are ready
        # for model fitting/export.
        _require_tensorflow()
        _configure_determinism(args.seed)

        features = _extract_dataset_features(raw_windows)
        expected_feature_shape = (args.window_size // 2, 1)
        if features.shape[1:] != expected_feature_shape:
            raise RuntimeError(
                "FFT feature extraction returned the wrong shape: "
                f"received {features.shape[1:]}, expected {expected_feature_shape}."
            )
        split = _stratified_split(
            features,
            labels,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed + 1,
        )

        model = _build_model(args.window_size // 2, args.learning_rate)
        history = _fit_model(
            model,
            split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        float_probabilities = np.asarray(
            model(split["test_features"], training=False).numpy(), dtype=np.float32
        ).reshape(-1)
        float_metrics = _binary_metrics(split["test_labels"], float_probabilities)

        model_content = _convert_to_int8(
            model,
            representative_features=split["train_features"],
            feature_bins=args.window_size // 2,
            representative_samples=args.representative_samples,
        )
        int8_probabilities, quantization = _validate_int8_tflite(
            model_content,
            split["test_features"],
            split["test_labels"],
            feature_bins=args.window_size // 2,
        )
        int8_metrics = _binary_metrics(split["test_labels"], int8_probabilities)

        model_sha256 = hashlib.sha256(model_content).hexdigest()
        metadata: Dict[str, Any] = {
            # Keep the existing metadata version for the synthetic/firmware
            # consumers; the new data provenance fields are additive.
            "schema_version": 1,
            "seed": int(args.seed),
            "data_mode": data_mode,
            "preprocessing": {
                "steps": [
                    "three-axis vector magnitude sqrt(x*x + y*y + z*z)",
                    "subtract the mean magnitude from the window",
                    "multiply by symmetric Hann: 0.5 - 0.5*cos(2*pi*n/(N-1))",
                    "real one-sided FFT (rfft)",
                    "drop the DC bin (bin 0)",
                    "divide FFT magnitude by N/2",
                    "log1p of FFT magnitude",
                    "per-window z-score",
                ],
                "output_shape": [int(args.window_size // 2), 1],
                "z_score_epsilon": _Z_SCORE_EPSILON,
                "fft_input_dtype": "float64 reference arithmetic, float32 model input",
            },
            "dataset": {
                "total_windows": int(features.shape[0]),
                "train_windows": int(split["train_features"].shape[0]),
                "validation_windows": int(split["validation_features"].shape[0]),
                "test_windows": int(split["test_features"].shape[0]),
                "test_fraction": float(args.test_fraction),
                "validation_fraction": float(args.validation_fraction),
                "class_counts": {
                    "normal": int(np.sum(labels == 0)),
                    "anomaly": int(np.sum(labels == 1)),
                },
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "shuffle": False,
                "history": history,
            },
            "model": {
                "name": "vibration_cnn",
                "architecture": "Conv1D(8,5)-MaxPool-Conv1D(12,3)-GlobalAveragePool-Dense(8)-Dense(1)",
                "output_activation": "sigmoid",
                "input_shape": [1, int(args.window_size // 2), 1],
                "output_shape": [1, 1],
                "input_dtype": "int8",
                "output_dtype": "int8",
            },
            "quantization": quantization,
            # Keep the two runtime contracts easy to discover for firmware
            # tooling that does not want to understand the full metadata tree.
            "input_quantization": quantization["input"],
            "output_quantization": quantization["output"],
            "metrics": {
                "float_model_test": float_metrics,
                "int8_tflite_test": int8_metrics,
            },
            "artifacts": {
                "tflite_path": str(tflite_path),
                "metadata_path": str(metadata_path),
                "header_path": str(header_path),
                "tflite_size_bytes": int(len(model_content)),
                "tflite_sha256": model_sha256,
            },
        }
        if data_mode == "synthetic":
            # Keep the original synthetic metadata contract intact when no
            # real-data arguments are supplied.
            metadata["simulation"] = {
                "sensor": "MPU6050 (synthetic three-axis vibration)",
                "sample_rate_hz": float(args.sample_rate_hz),
                "window_size": int(args.window_size),
                "axes": ["x", "y", "z"],
                "normal_windows": int(args.normal_windows),
                "anomaly_windows": int(args.anomaly_windows),
                "amplitude_units": "synthetic-g",
            }
            metadata["assumptions"] = [
                "Synthetic normal/anomaly labels are generated by the deterministic bring-up simulator, not field ground truth."
            ]
        else:
            if real_data_metadata is None:  # pragma: no cover - mode branch guard
                raise RuntimeError("Real-data metadata was not produced for real-data mode.")
            metadata["real_data"] = real_data_metadata
            # Keep bearing counts and calibration statistics easy to discover
            # for consumers that do not traverse the source tree.
            metadata["bearing"] = real_data_metadata["bearing"]
            metadata["assumptions"] = list(real_data_metadata["assumptions"])
        header_text = _render_model_header(
            model_content,
            input_shape=quantization["input"]["shape"],
            input_quantization=quantization["input"],
            output_quantization=quantization["output"],
        )
        _write_outputs(
            tflite_path=tflite_path,
            metadata_path=metadata_path,
            header_path=header_path,
            model_content=model_content,
            metadata=metadata,
            header_text=header_text,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"error: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print("Training and INT8 export completed.")
    print(f"  TFLite model: {tflite_path}")
    print(f"  Metadata JSON: {metadata_path}")
    print(f"  C++ header: {header_path}")
    print(f"  INT8 test accuracy: {int8_metrics['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
