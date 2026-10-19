# ESP32-S3 vibration predictive maintenance

This directory contains a two-part, reproducible edge-AI example for an
MPU6050 accelerometer:

```text
predictive_maintenance/
├── training/
│   └── train_vibration_model.py       # simulate, train, quantize, export
├── firmware/
│   ├── platformio.ini                  # ESP32-S3 Arduino project settings
│   ├── include/model.h                 # generated; not present until training
│   └── src/main.cpp                    # sensor, FFT, TFLM inference loop
├── dashboard/
│   └── index.html                       # live threshold and workflow simulation
├── data/
│   ├── kirloskar_vibration.csv          # raw motor vibration / normal replay
│   ├── bearing_features.csv             # labelled bearing feature rows
│   └── gearbox_b30hz40.csv              # extracted a1..a4 gearbox replay
└── artifacts/                          # generated .tflite and JSON metadata
```

The training script and firmware deliberately share one preprocessing contract:

1. Compute three-axis acceleration magnitude: `sqrt(x*x + y*y + z*z)`.
2. Subtract the window mean.
3. Multiply by the symmetric Hann window.
4. Compute a real, one-sided FFT.
5. Remove the DC bin and divide each magnitude by `N/2`.
6. Apply `log1p`.
7. Apply a per-window z-score.

The default contract is a 1024-sample window at 1 kHz, producing 512 features.
The generated model accepts `[1, 512, 1]` int8 features and returns one int8
sigmoid anomaly probability. If any of those values change, regenerate
`model.h` and rebuild the firmware together.

## Part 1: train and export

Create a virtual environment and install the host-side dependencies:

```bash
cd predictive_maintenance
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy>=1.23" "tensorflow>=2.13,<2.17"
```

On Apple Silicon, use the TensorFlow package supported by the installed Python
version (some environments use `tensorflow-macos` instead of `tensorflow`).
The script itself only imports NumPy and TensorFlow; no scikit-learn dependency
is required.

Run the deterministic synthetic-data example:

```bash
python training/train_vibration_model.py
```

The command writes:

- `artifacts/vibration_anomaly_int8.tflite`: fully integer TFLite model;
- `artifacts/vibration_anomaly_metrics.json`: preprocessing contract,
  quantization parameters, model metadata, and float/int8 test metrics;
- `firmware/include/model.h`: aligned C++ byte array (`g_model` and
  `g_model_len`) ready for the firmware;

You can make the simulation larger or use a different output directory:

```bash
python training/train_vibration_model.py \
  --normal-windows 1000 \
  --anomaly-windows 1000 \
  --epochs 25 \
  --artifacts-dir artifacts
```

The data generator is a bring-up aid, not a production fault model. Before
deploying, replace it with labeled, sensor-mounted data that covers machine
speed, load, mounting, temperature, sensor orientation, and the failure modes
that matter for the asset. Recalibrate the alert threshold on a held-out field
dataset; the training metrics use a `0.5` probability threshold, while the
deployment/dashboard default is `0.55` and should be recalibrated on field
data.

### Train from the supplied datasets

The supplied files are now available under `predictive_maintenance/data/` and
are mapped as follows:

- `kirloskar_vibration.csv`: raw `accel_x/accel_y/accel_z` motor vibration;
  source-labelled normal calibration/replay;
- `bearing_features.csv`: nine feature columns plus `fault`, including
  `Normal_1`, ball, inner-race, and outer-race labels; retained as feature-level
  calibration/provenance metadata;
- `gearbox_b30hz40.csv`: extracted from the supplied `b30hz40.csv.zip`, with
  raw `a1..a4` channels; `a1..a3` are source-labelled gearbox anomaly replay
  channels and `a4` is validated but excluded from the ESP32 three-axis model.

The raw streams do not contain an explicit ground-truth normal/fault label or a
sampling-rate field. Real-data mode therefore records these source labels as
bring-up assumptions; it is not a substitute for field labels or a verified
sampling clock. The bearing rows are not mixed into the FFT model because the
ESP32 model consumes 512 FFT bins from raw three-axis windows.

Run the real-data ingestion path with the fixed ESP32 contract (1024 samples,
1 kHz, 512 FFT bins):

```bash
python training/train_vibration_model.py \
  --real-data-dir data \
  --epochs 25 \
  --artifacts-dir artifacts
```

The loader validates the three schemas, extracts complete raw windows, records
bearing label counts and calibration statistics, and writes the provenance to
`artifacts/vibration_anomaly_metrics.json`. TensorFlow is still required for
the actual model fit/export.

## AXON·EDGE dashboard simulation

The browser dashboard is available at:

```text
predictive_maintenance/dashboard/index.html
```

Run it locally without a frontend build tool:

```bash
cd predictive_maintenance
python3 -m http.server 4174
```

Open <http://localhost:4174/dashboard/>. The dashboard currently runs in **simulation
mode** and provides the front-end integration boundary for the embedded
backend. It includes:

For a deterministic alert demonstration, open
<http://localhost:4174/dashboard/?scenario=bearing> or use the **Inject Fault** and
scenario buttons in the dashboard.

- live three-axis vibration and temperature plots;
- a healthy baseline learned at boot from the synthetic signal;
- a normal band for each of eight monitored features, calculated as
  `mean ± 2.5 × standard deviation`;
- a global anomaly threshold, defaulting to `0.55`, with a visible slider;
- hysteresis/debounce: the alert clears only after three windows below 80% of
  the threshold;
- relay state, anomaly score, health index, RUL projection, fleet cards, alert
  density, workflow stages, and an incident ledger;
- fault injection for imbalance, misalignment, bearing wear, outer-race
  impacts, and overheating;
- project-path dataset selectors for the Kirloskar, bearing-feature, and
  gearbox files under `data/`;
- CSV/TSV drop-zone intake with numeric-column mapping to X/Y/Z/temperature or
  the nine bearing feature columns;
- local CSV export and browser-local incident persistence;
- one selectable investigation tab per persisted incident, with alert-time score,
  threshold, feature-band, source/dataset, telemetry, relay, workflow, and event
  history details;
- JSON export for the selected incident investigation report.

The dashboard's dependency-free incident contract check can be run with:

```bash
cd predictive_maintenance/dashboard
python3 -m unittest -v test_incident_investigation
```

The dashboard threshold engine is intentionally a simulation adapter, not a
claim that the browser is executing the ESP32 TFLM binary. Once the model is
trained and `model.h` is generated, a later integration step can replace the
simulated score source with either serial/WebSocket/MQTT telemetry or a
browser-compatible inference service. The UI already exposes the same
threshold, normal-band, incident, and workflow concepts needed by that live
adapter.

## Part 2: build the ESP32-S3 firmware

The supplied `firmware/platformio.ini` uses the PlatformIO/Arduino-compatible
`ESP_TF` TFLM port, which packages Espressif's TFLM sources and ESP-NN kernels.
Install PlatformIO, generate `model.h` first, then build from the firmware
directory:

```bash
cd predictive_maintenance/firmware
pio run
pio run -t upload
pio device monitor
```

The only external PlatformIO library used by `main.cpp` is the ESP_TF
TensorFlow Lite Micro port:

```bash
pio pkg install --library "nickjgniklu/ESP_TF@^2.1.1"
```

`Wire` comes from Arduino-ESP32. The example does not require an additional
MPU6050 or ArduinoFFT library because it performs the MPU6050 register reads
and radix-2 FFT locally. Connect the sensor's SDA/SCL to the pins in
`platformio.ini`, share ground, and use a suitable regulated 3.3 V supply.
Override the pins by changing the two `build_flags` entries in
`platformio.ini` (or by adding equivalent `-DPM_MPU6050_SDA_PIN=<pin>` and
`-DPM_MPU6050_SCL_PIN=<pin>` flags in your board-specific PlatformIO
environment); no source edit is required.

The firmware checks the generated model's schema, input/output shapes, types,
and quantization metadata at boot. It collects a one-second window, performs
the local FFT, quantizes features with the TFLite input scale/zero point,
invokes the model, and prints for example:

```text
anomaly_confidence=0.083 status=NORMAL
```

### ESP-NN acceleration

ESP-NN is Espressif's optimized kernel library for ESP chips. The supplied
PlatformIO environment enables it for the ESP32-S3 with all three required
compile flags: `ESP_NN`, `CONFIG_NN_OPTIMIZED`, and `ARCH_ESP32_S3`. Those flags
are intentionally in `platformio.ini`, not only in `main.cpp`, because the
optimized kernels are compiled as library translation units. `ESP_TF` 2.1.1 or
newer is required for the ESP32-S3 kernel layout used here.

If you replace `ESP_TF` with another TFLM library, verify that it includes the
ESP-NN sources and supports those flags; adding the macros to an arbitrary
reference-only TFLM package will not create optimized kernels. The source
itself does not call an ESP-NN API, so the same `main.cpp` remains portable
between reference TFLM and an ESP-NN-enabled port. The 128 KiB arena is
intentionally conservative; after the exact model and TFLM build are known,
use the reported `arena_used_bytes()` value plus headroom to tune it.

## Hardware and deployment notes

- The default MPU6050 address is `0x68`; move AD0 high and update
  `kMpu6050Address` if the board uses `0x69`.
- The firmware configures the accelerometer for ±2 g and uses the raw
  register path. Add calibration and saturation/over-range handling before
  treating the score as a safety signal.
- The sensor is sampled at 1 kHz. The MPU6050's analog bandwidth and the
  machine's actual fault frequencies must be checked together; avoid relying
  on frequencies the sensor or mounting cannot capture.
- The generated C array is model data, not source code to hand-edit. Regenerate
  it whenever the model, FFT length, preprocessing, or quantization contract
  changes.
- If `AllocateTensors()` fails, first check that `model.h` came from the same
  training run, that the resolver contains every exported operator, and that
  the arena is large enough. The source includes a commented diagnostic path
  for temporarily trying `AllOpsResolver`; keep a selective resolver for the
  shipped build.
