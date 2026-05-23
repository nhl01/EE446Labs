#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino_BMI270_BMM150.h>
#include <math.h>

// These symbols are defined in the uploaded .cc model files.
// Keep the .cc files in the same Arduino sketch folder as this .ino file.
extern unsigned char encoder_clf_raw_pruned_qat_int8_tflite[];
extern unsigned int encoder_clf_raw_pruned_qat_int8_tflite_len;

extern unsigned char encoder_clf_std_pruned_qat_int8_tflite[];
extern unsigned int encoder_clf_std_pruned_qat_int8_tflite_len;

extern unsigned char encoder_clf_minmax_pruned_qat_int8_tflite[];
extern unsigned int encoder_clf_minmax_pruned_qat_int8_tflite_len;

extern unsigned char stacked_meta_clf_pruned_qat_int8_tflite[];
extern unsigned int stacked_meta_clf_pruned_qat_int8_tflite_len;

// Activity labels used by the trained models.
const char* label_map[12] = {
  "Standing still",
  "Sitting and relaxing",
  "Lying down",
  "Walking",
  "Climbing stairs",
  "Waist bends forward",
  "Frontal elevation of arms",
  "Knees bending (crouching)",
  "Cycling",
  "Jogging",
  "Running",
  "Jump front and back"
};

// Window and model dimensions.
const int kWindowSize = 100;
const int kFeatureCount = 6;
const int kInputSize = kWindowSize * kFeatureCount;
const int kNumClasses = 12;
const int kMetaInputSize = 3 * kNumClasses;

// Sampling configuration. 100 samples at 50 Hz gives a 2-second window.
const unsigned long kSampleIntervalMs = 20;

// Unit conversions for Arduino Nano 33 BLE Sense Rev2 IMU readings.
// Arduino_BMI270_BMM150 returns acceleration in g and gyroscope values in deg/s.
const float kGToMs2 = 9.80665f;
const float kDegToRad = 0.017453292519943295f;

// Tensor arena. Increase slightly if AllocateTensors fails.
const int kTensorArenaSize = 56 * 1024;
__attribute__((aligned(32), section(".noinit"))) static uint8_t tensor_arena[kTensorArenaSize];

// Data buffers.
__attribute__((aligned(32), section(".noinit"))) static float window_buffer[kWindowSize][kFeatureCount];
static float model_input[kInputSize];
static float raw_out[kNumClasses];
static float std_out[kNumClasses];
static float minmax_out[kNumClasses];
static float stacked_input[kMetaInputSize];
static float meta_out[kNumClasses];

// Normalization constants from the training pipeline.
float standard_means[6] = {3.268805f, -9.153526f, 0.700608f, -0.020571f, -0.520382f, -0.299817f};
float standard_stds[6]  = {4.253204f,  5.397565f, 6.433631f,  0.325429f,  0.571098f,  0.542104f};
float minmax_mins[6]    = {-22.091f, -19.571f, -19.363f, -0.977740f, -1.611600f, -1.206300f};
float minmax_maxs[6]    = { 20.003f,  20.909f,  24.599f,  1.463800f,  1.557200f,  1.326100f};

// TensorFlow Lite Micro objects.
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::AllOpsResolver resolver;

// Sampling state.
int sample_index = 0;
bool have_accel = false;
bool have_gyro = false;
float latest_ax = 0.0f;
float latest_ay = 0.0f;
float latest_az = 0.0f;
float latest_gx = 0.0f;
float latest_gy = 0.0f;
float latest_gz = 0.0f;
unsigned long last_sample_time = 0;
unsigned long last_status_time = 0;

int8_t QuantizeToInt8(float value, float scale, int zero_point) {
  int quantized = (int)lroundf(value / scale) + zero_point;
  if (quantized > 127) quantized = 127;
  if (quantized < -128) quantized = -128;
  return (int8_t)quantized;
}

bool RunInt8Model(const unsigned char* model_bytes,
                  const char* model_name,
                  const float* input_data,
                  int input_length,
                  float* output_data,
                  int output_length) {
  const tflite::Model* model = tflite::GetModel(model_bytes);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch for ");
    Serial.println(model_name);
    return false;
  }

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.print("AllocateTensors failed for ");
    Serial.println(model_name);
    return false;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    Serial.print("Expected int8 input/output for ");
    Serial.println(model_name);
    return false;
  }

  if ((int)input->bytes < input_length) {
    Serial.print("Input tensor is smaller than expected for ");
    Serial.println(model_name);
    return false;
  }

  if ((int)output->bytes < output_length) {
    Serial.print("Output tensor is smaller than expected for ");
    Serial.println(model_name);
    return false;
  }

  for (int i = 0; i < input_length; i++) {
    input->data.int8[i] = QuantizeToInt8(input_data[i], input->params.scale, input->params.zero_point);
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    Serial.print("Inference failed for ");
    Serial.println(model_name);
    return false;
  }

  for (int i = 0; i < output_length; i++) {
    output_data[i] = ((float)output->data.int8[i] - (float)output->params.zero_point) * output->params.scale;
  }

  return true;
}

void BuildInputFromWindow(int mode) {
  // mode = 0: raw input
  // mode = 1: standard-scaled input
  // mode = 2: min-max-scaled input
  for (int i = 0; i < kWindowSize; i++) {
    for (int j = 0; j < kFeatureCount; j++) {
      const int idx = i * kFeatureCount + j;
      const float value = window_buffer[i][j];

      if (mode == 0) {
        model_input[idx] = value;
      } else if (mode == 1) {
        model_input[idx] = (value - standard_means[j]) / standard_stds[j];
      } else {
        model_input[idx] = (value - minmax_mins[j]) / (minmax_maxs[j] - minmax_mins[j]);
      }
    }
  }
}

void PrintClassScores(const char* title, const float* scores, int length) {
  Serial.print(title);
  Serial.print(": ");
  for (int i = 0; i < length; i++) {
    Serial.print(scores[i], 4);
    if (i < length - 1) Serial.print(", ");
  }
  Serial.println();
}

bool RunEnsembleInference() {
  Serial.println("Window collected. Running ensemble inference...");

  BuildInputFromWindow(0);
  if (!RunInt8Model(encoder_clf_raw_pruned_qat_int8_tflite, "raw encoder-classifier", model_input, kInputSize, raw_out, kNumClasses)) {
    return false;
  }

  BuildInputFromWindow(1);
  if (!RunInt8Model(encoder_clf_std_pruned_qat_int8_tflite, "standard-scaled encoder-classifier", model_input, kInputSize, std_out, kNumClasses)) {
    return false;
  }

  BuildInputFromWindow(2);
  if (!RunInt8Model(encoder_clf_minmax_pruned_qat_int8_tflite, "min-max-scaled encoder-classifier", model_input, kInputSize, minmax_out, kNumClasses)) {
    return false;
  }

  for (int i = 0; i < kNumClasses; i++) {
    stacked_input[i] = raw_out[i];
    stacked_input[i + kNumClasses] = std_out[i];
    stacked_input[i + 2 * kNumClasses] = minmax_out[i];
  }

  if (!RunInt8Model(stacked_meta_clf_pruned_qat_int8_tflite, "stacked meta-classifier", stacked_input, kMetaInputSize, meta_out, kNumClasses)) {
    return false;
  }

  PrintClassScores("Raw model scores", raw_out, kNumClasses);
  PrintClassScores("Standard-scaled model scores", std_out, kNumClasses);
  PrintClassScores("Min-max-scaled model scores", minmax_out, kNumClasses);
  PrintClassScores("Meta-classifier scores", meta_out, kNumClasses);

  int best_class = 0;
  float best_score = meta_out[0];
  for (int i = 1; i < kNumClasses; i++) {
    if (meta_out[i] > best_score) {
      best_score = meta_out[i];
      best_class = i;
    }
  }

  Serial.println("----------------------------------------");
  Serial.print("Final prediction: ");
  Serial.println(label_map[best_class]);
  Serial.print("Class index: ");
  Serial.println(best_class);
  Serial.print("Confidence score: ");
  Serial.println(best_score, 4);
  Serial.println("----------------------------------------");

  return true;
}

void setup() {
  Serial.begin(115200);
  const unsigned long serial_start = millis();
  while (!Serial && (millis() - serial_start < 5000)) {
    // Wait briefly for Serial Monitor, but do not block forever.
  }

  Serial.println();
  Serial.println("Tiny Ensemble Learning inference sketch");

  if (!IMU.begin()) {
    Serial.println("IMU initialization failed. Check the board and selected IMU library.");
    while (1) {
      delay(1000);
    }
  }

  Serial.println("IMU initialized. Collecting 100-sample windows at 50 Hz.");
}

void loop() {
  // Read the latest accelerometer and gyroscope values independently.
  // This is more reliable than requiring both sensors to become available in the same loop iteration.
  if (IMU.accelerationAvailable()) {
    float ax, ay, az;
    if (IMU.readAcceleration(ax, ay, az)) {
      latest_ax = ax * kGToMs2;
      latest_ay = ay * kGToMs2;
      latest_az = az * kGToMs2;
      have_accel = true;
    }
  }

  if (IMU.gyroscopeAvailable()) {
    float gx, gy, gz;
    if (IMU.readGyroscope(gx, gy, gz)) {
      latest_gx = gx * kDegToRad;
      latest_gy = gy * kDegToRad;
      latest_gz = gz * kDegToRad;
      have_gyro = true;
    }
  }

  if (!have_accel || !have_gyro) {
    if (millis() - last_status_time > 1000) {
      Serial.println("Waiting for accelerometer and gyroscope data...");
      last_status_time = millis();
    }
    return;
  }

  if (millis() - last_sample_time < kSampleIntervalMs) {
    return;
  }
  last_sample_time = millis();

  window_buffer[sample_index][0] = latest_ax;
  window_buffer[sample_index][1] = latest_ay;
  window_buffer[sample_index][2] = latest_az;
  window_buffer[sample_index][3] = latest_gx;
  window_buffer[sample_index][4] = latest_gy;
  window_buffer[sample_index][5] = latest_gz;

  sample_index++;

  if (sample_index % 25 == 0) {
    Serial.print("Samples collected: ");
    Serial.println(sample_index);
  }

  if (sample_index < kWindowSize) {
    return;
  }

  sample_index = 0;

  if (!RunEnsembleInference()) {
    Serial.println("Inference failed. Stopping execution for debugging.");
    while (1) {
      delay(1000);
    }
  }
}
