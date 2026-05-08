#include "TensorFlowLite.h"
#include "autoencoder_model.cc"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include <Arduino_BMI270_BMM150.h>
#include <math.h>

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Constants
const int kWindowSize = 100;
const int kInputSize = 300;  // 100 time steps x 3 axes
const int kTensorArenaSize = 64 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Accelerometer buffer
float window_buffer[kWindowSize][3];
int sample_index = 0;
bool window_filled = false;

// TFLite variables
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Quantization parameters
float input_scale;
int input_zero_point;
float output_scale;
int output_zero_point;

// Threshold
const float kReconstructionErrorThreshold = 2;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1);
  }

  Serial.println("IMU initialized.");

  const tflite::Model* model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch.");
    while (1);
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    error_reporter
  );

  interpreter = &static_interpreter;

  Serial.println("Allocating tensors.");

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed.");
    while (1);
  }

  Serial.println("AllocateTensors successful.");

  input = interpreter->input(0);
  output = interpreter->output(0);

  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;
  output_scale = output->params.scale;
  output_zero_point = output->params.zero_point;

  Serial.println("Model setup complete.");
}

void loop() {
  float x, y, z;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);

    window_buffer[sample_index][0] = x;
    window_buffer[sample_index][1] = y;
    window_buffer[sample_index][2] = z;

    sample_index++;

    if (sample_index >= kWindowSize) {
      sample_index = 0;
      window_filled = true;
    }

    if (window_filled) {
      for (int i = 0; i < kWindowSize; i++) {
        for (int j = 0; j < 3; j++) {
          float val = window_buffer[i][j];
          int index = i * 3 + j;

          int32_t q = lround(val / input_scale) + input_zero_point;
          q = constrain(q, -128, 127);

          input->data.int8[index] = static_cast<int8_t>(q);
        }
      }

      if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed.");
        return;
      }

      float recon_error = 0.0;

      for (int i = 0; i < kInputSize; i++) {
        float original = window_buffer[i / 3][i % 3];

        int8_t quant_pred = output->data.int8[i];
        float predicted = (quant_pred - output_zero_point) * output_scale;

        float diff = original - predicted;
        recon_error += diff * diff;
      }

      recon_error /= kInputSize;

      Serial.print("Reconstruction error: ");
      Serial.println(recon_error, 6);

      if (recon_error > kReconstructionErrorThreshold) {
        Serial.println("Result: Anomaly detected");
      } else {
        Serial.println("Result: Normal activity");
      }
    }
  }

  delay(20);  // Approximately 50 Hz sampling
}