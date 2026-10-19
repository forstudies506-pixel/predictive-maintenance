// ESP32-S3 predictive-maintenance edge inference.
//
// The generated model.h is expected to be produced from the int8 TFLite model
// with the usual xxd -i symbols:
//   extern const unsigned char g_model[];
//   extern const int g_model_len;
//
// This file deliberately keeps the sensor path local: Wire is used directly
// instead of depending on an MPU6050 library, and the FFT is implemented here
// so that its preprocessing is auditable against the Python backend.

#include <Arduino.h>
#include <Wire.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "model.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// GPIO defaults are common on ESP32-S3 development boards, but board wiring
// wins. Override these in PlatformIO build_flags when the board routes I2C
// elsewhere, for example -DPM_MPU6050_SDA_PIN=4.
#ifndef PM_MPU6050_SDA_PIN
#define PM_MPU6050_SDA_PIN 8
#endif

#ifndef PM_MPU6050_SCL_PIN
#define PM_MPU6050_SCL_PIN 9
#endif

namespace {

// These constants are part of the Python/edge feature contract. Keep them in
// sync with the training/export code: one 1024-sample, 1 kHz window produces
// 512 one-sided FFT bins.
constexpr size_t kFftSize = 1024;
constexpr size_t kFeatureBins = kFftSize / 2;
constexpr uint32_t kSampleRateHz = 1000;
constexpr uint32_t kSamplePeriodUs = 1000000UL / kSampleRateHz;
constexpr float kPi = 3.14159265358979323846f;
// Field threshold is intentionally a deployment setting, not a model weight.
// The dashboard simulation uses the same default and exposes it as a control.
constexpr float kAnomalyThreshold = 0.55f;

static_assert(kFftSize == 1024, "The edge FFT must match the backend window");
static_assert(kFeatureBins == 512,
              "The edge feature vector must match the backend bins");

constexpr uint8_t kMpu6050Address = 0x68;
constexpr uint8_t kMpu6050PwrMgmt1 = 0x6B;
constexpr uint8_t kMpu6050Config = 0x1A;
constexpr uint8_t kMpu6050SampleRateDivider = 0x19;
constexpr uint8_t kMpu6050AccelConfig = 0x1C;
constexpr uint8_t kMpu6050GyroConfig = 0x1B;
constexpr uint8_t kMpu6050AccelData = 0x3B;

constexpr int kMpuSdaPin = PM_MPU6050_SDA_PIN;
constexpr int kMpuSclPin = PM_MPU6050_SCL_PIN;

// The arena is intentionally static and 16-byte aligned. ESP32-S3 has useful
// internal SRAM, but keeping the arena out of the task stack avoids a large
// stack reservation. Measure interpreter->arena_used_bytes() on the target
// and reduce this value only after the exact generated model is known.
// Some ESP32-S3 TFLM packages route supported int8 Conv/FC kernels through
// ESP-NN; this file does not include or assume an ESP-NN API, so acceleration
// remains package/toolchain-controlled and the same code also works with the
// reference kernels.
constexpr size_t kTensorArenaSize = 128 * 1024;
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

// These buffers are also static: 1024 floats for the sample window, two 1024
// element FFT work arrays, and 512 feature floats are several kilobytes and
// should not be placed on the Arduino loopTask stack.
static float g_sample_magnitude[kFftSize];
static float g_hann_window[kFftSize];
static float g_fft_real[kFftSize];
static float g_fft_imag[kFftSize];
static float g_features[kFeatureBins];

static const tflite::Model* g_model_view = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input = nullptr;
static TfLiteTensor* g_output = nullptr;
static bool g_ready = false;

void PrintError(const char* message) {
  Serial.print("ERROR: ");
  Serial.println(message);
}

bool WriteMpuRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(kMpu6050Address);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission(true) == 0;
}

bool ReadMpuRegisters(uint8_t reg, uint8_t* destination, size_t length) {
  if (destination == nullptr || length == 0 || length > 255) {
    return false;
  }

  Wire.beginTransmission(kMpu6050Address);
  Wire.write(reg);
  // A repeated-start keeps this register read on the same I2C transaction.
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  const size_t received = Wire.requestFrom(
      static_cast<uint8_t>(kMpu6050Address), static_cast<uint8_t>(length),
      static_cast<uint8_t>(true));
  if (received != length) {
    while (Wire.available() > 0) {
      (void)Wire.read();
    }
    return false;
  }

  for (size_t i = 0; i < length; ++i) {
    if (Wire.available() <= 0) {
      return false;
    }
    destination[i] = static_cast<uint8_t>(Wire.read());
  }
  return true;
}

bool ConfigureMpu6050() {
  // Wake the device, select a 1 kHz accelerometer output path, and select the
  // +/-2 g accelerometer range (16384 LSB/g). The sample loop below still
  // enforces the 1 kHz collection period rather than relying on FIFO timing.
  // With DLPF_CFG=0 the MPU6050 internal sample clock is 8 kHz; divider 7
  // makes the register update rate 8 kHz / (7 + 1) = 1 kHz.
  if (!WriteMpuRegister(kMpu6050PwrMgmt1, 0x00) ||
      !WriteMpuRegister(kMpu6050Config, 0x00) ||
      !WriteMpuRegister(kMpu6050SampleRateDivider, 0x07) ||
      !WriteMpuRegister(kMpu6050AccelConfig, 0x00) ||
      !WriteMpuRegister(kMpu6050GyroConfig, 0x00)) {
    return false;
  }

  // A short read verifies that the raw I2C path is alive without adding a
  // sensor-library dependency. The MPU6050 WHO_AM_I register is 0x75.
  uint8_t who_am_i = 0;
  if (!ReadMpuRegisters(0x75, &who_am_i, 1)) {
    return false;
  }
  return who_am_i == kMpu6050Address;
}

bool ReadMpu6050VectorMagnitude(float* magnitude) {
  if (magnitude == nullptr) {
    return false;
  }

  uint8_t raw[6] = {};
  if (!ReadMpuRegisters(kMpu6050AccelData, raw, sizeof(raw))) {
    return false;
  }

  const int16_t ax = static_cast<int16_t>(
      (static_cast<uint16_t>(raw[0]) << 8) | raw[1]);
  const int16_t ay = static_cast<int16_t>(
      (static_cast<uint16_t>(raw[2]) << 8) | raw[3]);
  const int16_t az = static_cast<int16_t>(
      (static_cast<uint16_t>(raw[4]) << 8) | raw[5]);

  // Scaling to g is not required by z-score normalization, but makes the
  // intermediate value inspectable and documents the configured range.
  constexpr float kAccelLsbPerG = 16384.0f;
  const float ax_g = static_cast<float>(ax) / kAccelLsbPerG;
  const float ay_g = static_cast<float>(ay) / kAccelLsbPerG;
  const float az_g = static_cast<float>(az) / kAccelLsbPerG;
  *magnitude = sqrtf(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
  return isfinite(*magnitude);
}

bool CollectWindow(float* magnitude) {
  if (magnitude == nullptr) {
    return false;
  }

  uint32_t next_sample_us = micros();
  for (size_t i = 0; i < kFftSize; ++i) {
    // Signed subtraction remains correct across the micros() wrap point.
    while (static_cast<int32_t>(micros() - next_sample_us) < 0) {
      delayMicroseconds(20);
    }

    if (!ReadMpu6050VectorMagnitude(&magnitude[i])) {
      return false;
    }
    next_sample_us += kSamplePeriodUs;
  }
  return true;
}

void InitializeHannWindow() {
  // Compute this once instead of evaluating cosf() for every sample in every
  // inference window. The formula is the same symmetric Hann definition used
  // by the Python training backend.
  for (size_t i = 0; i < kFftSize; ++i) {
    g_hann_window[i] =
        0.5f * (1.0f - cosf(2.0f * kPi * static_cast<float>(i) /
                             static_cast<float>(kFftSize - 1)));
  }
}

// In-place radix-2 Cooley-Tukey FFT. A real N-point FFT has N/2+1 one-sided
// bins. The Python backend drops bin 0 (DC), so the exported 512 bins are
// FFT bins 1..N/2 inclusive. The divisor below is exactly N/2, matching the
// edge contract's magnitude/(N/2) convention.
void Radix2RealFftMagnitude(const float* real_input, float* magnitude) {
  for (size_t i = 0; i < kFftSize; ++i) {
    g_fft_real[i] = real_input[i];
    g_fft_imag[i] = 0.0f;
  }

  // Bit-reversal permutation.
  for (size_t i = 1, j = 0; i < kFftSize; ++i) {
    size_t bit = kFftSize >> 1;
    while ((j & bit) != 0) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const float real_tmp = g_fft_real[i];
      g_fft_real[i] = g_fft_real[j];
      g_fft_real[j] = real_tmp;
      const float imag_tmp = g_fft_imag[i];
      g_fft_imag[i] = g_fft_imag[j];
      g_fft_imag[j] = imag_tmp;
    }
  }

  for (size_t length = 2; length <= kFftSize; length <<= 1) {
    const float angle = -2.0f * kPi / static_cast<float>(length);
    const float step_real = cosf(angle);
    const float step_imag = sinf(angle);
    const size_t half_length = length >> 1;

    for (size_t start = 0; start < kFftSize; start += length) {
      float twiddle_real = 1.0f;
      float twiddle_imag = 0.0f;
      for (size_t offset = 0; offset < half_length; ++offset) {
        const size_t even = start + offset;
        const size_t odd = even + half_length;
        const float odd_real = g_fft_real[odd] * twiddle_real -
                               g_fft_imag[odd] * twiddle_imag;
        const float odd_imag = g_fft_real[odd] * twiddle_imag +
                               g_fft_imag[odd] * twiddle_real;

        const float even_real = g_fft_real[even];
        const float even_imag = g_fft_imag[even];
        g_fft_real[even] = even_real + odd_real;
        g_fft_imag[even] = even_imag + odd_imag;
        g_fft_real[odd] = even_real - odd_real;
        g_fft_imag[odd] = even_imag - odd_imag;

        const float next_twiddle_real = twiddle_real * step_real -
                                        twiddle_imag * step_imag;
        twiddle_imag = twiddle_real * step_imag +
                       twiddle_imag * step_real;
        twiddle_real = next_twiddle_real;
      }
    }
  }

  constexpr float kMagnitudeDivisor = static_cast<float>(kFftSize / 2);
  for (size_t bin = 0; bin < kFeatureBins; ++bin) {
    const size_t fft_bin = bin + 1;  // Drop the DC bin, as in the backend.
    const float power = g_fft_real[fft_bin] * g_fft_real[fft_bin] +
                        g_fft_imag[fft_bin] * g_fft_imag[fft_bin];
    magnitude[bin] = sqrtf(power) / kMagnitudeDivisor;
  }
}

// Local FFT preprocessing hook. This implementation is intentionally kept in
// one function so it can later be replaced by ESP-DSP/another vetted FFT
// kernel without changing the sensor collection or TFLM code around it.
// Exact local feature pipeline:
// vector magnitude -> remove mean -> Hann window -> real FFT ->
// magnitude/(N/2) -> log1p -> per-window z-score.
void BuildFeatures(const float* vector_magnitude, float* features) {
  float sample_mean = 0.0f;
  for (size_t i = 0; i < kFftSize; ++i) {
    sample_mean += vector_magnitude[i];
  }
  sample_mean /= static_cast<float>(kFftSize);

  // Reuse g_fft_real as a temporary windowed time-domain input. It is copied
  // into the FFT work array by Radix2RealFftMagnitude below.
  for (size_t i = 0; i < kFftSize; ++i) {
    g_fft_real[i] = (vector_magnitude[i] - sample_mean) * g_hann_window[i];
  }

  Radix2RealFftMagnitude(g_fft_real, features);

  float log_mean = 0.0f;
  for (size_t bin = 0; bin < kFeatureBins; ++bin) {
    features[bin] = log1pf(features[bin]);
    log_mean += features[bin];
  }
  log_mean /= static_cast<float>(kFeatureBins);

  float variance = 0.0f;
  for (size_t bin = 0; bin < kFeatureBins; ++bin) {
    const float centered = features[bin] - log_mean;
    variance += centered * centered;
  }
  variance /= static_cast<float>(kFeatureBins);

  const float standard_deviation = sqrtf(variance);
  // Match the backend's _Z_SCORE_EPSILON guard for nearly constant windows.
  if (standard_deviation <= 1.0e-6f || !isfinite(standard_deviation)) {
    for (size_t bin = 0; bin < kFeatureBins; ++bin) {
      features[bin] = 0.0f;
    }
    return;
  }

  for (size_t bin = 0; bin < kFeatureBins; ++bin) {
    features[bin] = (features[bin] - log_mean) / standard_deviation;
  }
}

size_t TensorElementCount(const TfLiteTensor* tensor) {
  if (tensor == nullptr || tensor->dims == nullptr || tensor->dims->size <= 0) {
    return 0;
  }

  size_t count = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    const int dimension = tensor->dims->data[i];
    if (dimension <= 0) {
      return 0;
    }
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

bool HasExpectedInputShape(const TfLiteTensor* tensor) {
  return tensor != nullptr && tensor->dims != nullptr &&
         tensor->dims->size == 3 && tensor->dims->data[0] == 1 &&
         tensor->dims->data[1] == static_cast<int>(kFeatureBins) &&
         tensor->dims->data[2] == 1;
}

bool HasExpectedOutputShape(const TfLiteTensor* tensor) {
  return tensor != nullptr && tensor->dims != nullptr &&
         tensor->dims->size == 2 && tensor->dims->data[0] == 1 &&
         tensor->dims->data[1] == 1;
}

bool QuantizeFeaturesIntoInput(const float* features, TfLiteTensor* input) {
  if (features == nullptr || input == nullptr || input->type != kTfLiteInt8 ||
      input->data.int8 == nullptr || input->params.scale <= 0.0f ||
      !isfinite(input->params.scale) || !HasExpectedInputShape(input)) {
    return false;
  }

  const float scale = input->params.scale;
  const int32_t zero_point = input->params.zero_point;
  for (size_t bin = 0; bin < kFeatureBins; ++bin) {
    const float real_value = features[bin];
    if (!isfinite(real_value)) {
      return false;
    }

    int32_t quantized = static_cast<int32_t>(
        roundf(real_value / scale) + static_cast<float>(zero_point));
    if (quantized < -128) {
      quantized = -128;
    } else if (quantized > 127) {
      quantized = 127;
    }
    input->data.int8[bin] = static_cast<int8_t>(quantized);
  }
  return true;
}

bool InitializeModel() {
  if (g_model_len <= 0) {
    PrintError("generated model is empty");
    return false;
  }

  g_model_view = tflite::GetModel(g_model);
  if (g_model_view == nullptr) {
    PrintError("GetModel returned null");
    return false;
  }

  // This checks the flatbuffer schema version expected by the linked TFLM
  // headers before any tensor metadata is trusted.
  Serial.print("model schema version: ");
  Serial.println(static_cast<int>(g_model_view->version()));
  if (g_model_view->version() != TFLITE_SCHEMA_VERSION) {
    PrintError("model schema version does not match TFLM");
    return false;
  }
  if (g_model_view->subgraphs() == nullptr ||
      g_model_view->subgraphs()->size() != 1) {
    PrintError("expected exactly one model subgraph");
    return false;
  }
  const tflite::SubGraph* subgraph = g_model_view->subgraphs()->Get(0);
  if (subgraph == nullptr || subgraph->inputs() == nullptr ||
      subgraph->inputs()->size() != 1 || subgraph->outputs() == nullptr ||
      subgraph->outputs()->size() != 1) {
    PrintError("expected exactly one model input and one output");
    return false;
  }

  // This is deliberately selective. Conv1D is represented by CONV_2D in
  // standard TFLite export; GlobalAveragePooling1D is commonly MEAN, while
  // some converter versions use AVERAGE_POOL_2D plus shape plumbing.
  constexpr unsigned int kResolverCapacity = 10;
  static tflite::MicroMutableOpResolver<kResolverCapacity> resolver;
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddAveragePool2D() != kTfLiteOk ||
      resolver.AddMean() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddLogistic() != kTfLiteOk ||
      resolver.AddReshape() != kTfLiteOk ||
      resolver.AddSqueeze() != kTfLiteOk ||
      resolver.AddExpandDims() != kTfLiteOk) {
    PrintError("failed to register the selective model ops");
    return false;
  }

  // If a converter emits an unexpected op, temporarily replacing the
  // resolver above with tflite::AllOpsResolver (from
  // tensorflow/lite/micro/all_ops_resolver.h) is a diagnostic fallback. It is
  // intentionally not the default: AllOpsResolver costs substantially more
  // flash/arena memory and should only be kept after inspecting the flatbuffer.
  static tflite::MicroInterpreter interpreter(
      g_model_view, resolver, g_tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    PrintError("AllocateTensors failed; check model ops and arena size");
    return false;
  }

  g_input = interpreter.input(0);
  g_output = interpreter.output(0);
  if (g_input == nullptr || g_output == nullptr) {
    PrintError("model input/output tensor is missing");
    return false;
  }

  if (g_input->type != kTfLiteInt8 ||
      !HasExpectedInputShape(g_input) ||
      TensorElementCount(g_input) != kFeatureBins ||
      g_input->data.int8 == nullptr || g_input->bytes < kFeatureBins ||
      g_input->params.scale <= 0.0f || !isfinite(g_input->params.scale)) {
    PrintError("input must be a quantized int8 tensor with 512 elements");
    return false;
  }
  if (g_output->type != kTfLiteInt8 || !HasExpectedOutputShape(g_output) ||
      TensorElementCount(g_output) != 1 ||
      g_output->data.int8 == nullptr || g_output->params.scale <= 0.0f ||
      !isfinite(g_output->params.scale)) {
    PrintError("output must be one quantized int8 sigmoid value");
    return false;
  }

  Serial.print("tensor arena used: ");
  Serial.println(static_cast<unsigned long>(interpreter.arena_used_bytes()));
  Serial.print("input scale/zero_point: ");
  Serial.print(g_input->params.scale, 8);
  Serial.print("/");
  Serial.println(g_input->params.zero_point);
  Serial.print("output scale/zero_point: ");
  Serial.print(g_output->params.scale, 8);
  Serial.print("/");
  Serial.println(g_output->params.zero_point);

  g_interpreter = &interpreter;
  return true;
}

bool RunInference(float* confidence) {
  if (confidence == nullptr || g_interpreter == nullptr || g_input == nullptr ||
      g_output == nullptr) {
    return false;
  }

  BuildFeatures(g_sample_magnitude, g_features);
  if (!QuantizeFeaturesIntoInput(g_features, g_input)) {
    return false;
  }
  if (g_interpreter->Invoke() != kTfLiteOk) {
    return false;
  }

  const int8_t output_value = g_output->data.int8[0];
  const float dequantized =
      (static_cast<int32_t>(output_value) - g_output->params.zero_point) *
      g_output->params.scale;
  if (!isfinite(dequantized)) {
    return false;
  }

  // Sigmoid should already be in [0, 1], but clamp the dequantized value so a
  // quantization endpoint roundoff cannot print an invalid confidence.
  *confidence = dequantized;
  if (*confidence < 0.0f) {
    *confidence = 0.0f;
  } else if (*confidence > 1.0f) {
    *confidence = 1.0f;
  }
  return true;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  const uint32_t serial_wait_start = millis();
  while (!Serial && (millis() - serial_wait_start) < 2000UL) {
    delay(10);
  }

  Serial.println();
  Serial.println("predictive maintenance edge starting");
  Serial.print("FFT/window: ");
  Serial.print(static_cast<unsigned long>(kFftSize));
  Serial.print(" samples at ");
  Serial.print(static_cast<unsigned long>(kSampleRateHz));
  Serial.println(" Hz; feature bins: 512");

  Wire.begin(kMpuSdaPin, kMpuSclPin);
  Wire.setClock(400000);
  if (!ConfigureMpu6050()) {
    PrintError("MPU6050 setup/read path failed");
    return;
  }
  Serial.println("MPU6050 ready");

  InitializeHannWindow();

  g_ready = InitializeModel();
  if (g_ready) {
    Serial.println("TFLM model ready");
  }
}

void loop() {
  if (!g_ready) {
    delay(1000);
    return;
  }

  if (!CollectWindow(g_sample_magnitude)) {
    PrintError("MPU6050 sample window failed");
    delay(100);
    return;
  }

  float confidence = 0.0f;
  if (!RunInference(&confidence)) {
    PrintError("TFLM inference failed");
    delay(100);
    return;
  }

  Serial.print("anomaly_confidence=");
  Serial.print(confidence, 3);
  Serial.print(" status=");
  Serial.println(confidence >= kAnomalyThreshold ? "ANOMALY" : "NORMAL");
}
