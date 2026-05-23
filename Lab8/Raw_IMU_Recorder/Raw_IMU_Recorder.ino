#include <Arduino_BMI270_BMM150.h>

// Sampling Parameters
const unsigned long kDurationMillis = 2UL * 60UL * 1000UL;  // 2 minutes
const int kSamplingRateHz = 50;  // 50Hz
const unsigned long kIntervalMillis = 1000 / kSamplingRateHz;

unsigned long start_time = 0;
unsigned long last_sample_time = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("❌ Failed to initialize IMU!");
    while (1);
  }

  Serial.println("✅ IMU initialized. Starting recording for 2 minutes...");
  Serial.println("timestamp_ms,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z");

  start_time = millis();
  last_sample_time = start_time;
}

void loop() {
  unsigned long current_time = millis();

  if (current_time - start_time >= kDurationMillis) {
    Serial.println("✅ Recording complete. You can now copy the CSV from Serial Monitor.");
    while (1);  // Stop execution
  }

  if (current_time - last_sample_time >= kIntervalMillis) {
    float ax, ay, az, gx, gy, gz;

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() &&
        IMU.readAcceleration(ax, ay, az) && IMU.readGyroscope(gx, gy, gz)) {

      Serial.print(current_time);
      Serial.print(",");
      Serial.print(ax, 6);
      Serial.print(",");
      Serial.print(ay, 6);
      Serial.print(",");
      Serial.print(az, 6);
      Serial.print(",");
      Serial.print(gx, 6);
      Serial.print(",");
      Serial.print(gy, 6);
      Serial.print(",");
      Serial.println(gz, 6);
    }

    last_sample_time = current_time;
  }
}
