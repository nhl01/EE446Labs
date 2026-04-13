#include <PDM.h>
#include <Arduino_APDS9960.h>
#include <Arduino_BMI270_BMM150.h>

const unsigned long SAMPLE_INTERVAL = 100;
unsigned long lastSampleTime = 0;

short sampleBuffer[256];
volatile int samplesRead = 0;

//Measurements
int level;
float x, y, z;
int r, g, b, c;
int proximity = -1;  //Initialize to invalid value

int sound, dark, moving, near = 0;  //flags
String final_label = "";

//Threshold values
const int NOISE_THRESHOLD = 100;
const int BRIGHTNESS_THRESHOLD = 500;
const float NOMINAL_ACCEL_MAGNITUDE = 1.01;
const float MOVING_THRESHOLD = 0.05;
const int PROX_THRESHOLD = 0;

void onPDMData() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  PDM.onReceive(onPDMData);

  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM microphone");
    while (1)
      ;
  }

  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1)
      ;
  }

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1)
      ;
  }
}

void loop() {
  unsigned long currTime = millis();  //Non-blocking "delay"
  if (currTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currTime;

    //IMU sensor
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x, y, z);
    }
    //Microphone
    if (samplesRead) {
      long sum = 0;
      for (int i = 0; i < samplesRead; i++) {
        sum += abs(sampleBuffer[i]);
      }
      level = sum / samplesRead;  //Microphone level
      samplesRead = 0;
    }

    //Light/proximity sensor
    if (APDS.colorAvailable() && APDS.proximityAvailable()) {  //Color takes the longest time, so print as soon as color data is ready. Also only print when proximity data is updated
      APDS.readColor(r, g, b, c);
      proximity = APDS.readProximity();
      calc_flags();
      setFinalLabel();
      Serial.print("raw, mic=");
      Serial.print(level);
      Serial.print(", clear=");
      Serial.print(c);
      Serial.print(", motion=<");
      Serial.print(x, 3);
      Serial.print(", ");
      Serial.print(y, 3);
      Serial.print(", ");
      Serial.print(z, 3);
      Serial.print(">, prox=");
      Serial.println(proximity);

      Serial.print("flags, sound=");
      Serial.print(sound);
      Serial.print(", dark=");
      Serial.print(dark);
      Serial.print(", moving=");
      Serial.print(moving);
      Serial.print(", near=");
      Serial.println(near);

      Serial.print("state, ");
      Serial.println(final_label);
    }
  }
}

float calc_motion() {  //Magnitude of acceleration vector
  return sqrt(x * x + y * y + z * z);
}

float calc_flags() {
  if (level > NOISE_THRESHOLD) sound = 1;
  else sound = 0;

  if (c < BRIGHTNESS_THRESHOLD) dark = 1;
  else dark = 0;

  if (abs(calc_motion() - NOMINAL_ACCEL_MAGNITUDE) > MOVING_THRESHOLD) moving = 1;  //If acceleration vector is larger than the still magnitude, then the board's moving
  else moving = 0;

  if (proximity > PROX_THRESHOLD) near = 0;
  else near = 1;
}

void setFinalLabel() {
  if (!sound && !dark && !moving && !near) {  //Flags should be 0000
    final_label = "QUIET_BRIGHT_STEADY_FAR";
  } else if (sound && !dark && !moving && !near) {  //Flags should be 1000
    final_label = "NOISY_BRIGHT_STEADY_FAR";
  } else if (!sound && dark && !moving && near) {  //Flags should be 0101
    final_label = "QUIET_DARK_STEADY_NEAR";
  } else if (sound && !dark && moving && near) {  //Flags should be 1011
    final_label = "NOISY_BRIGHT_MOVING_NEAR";
  } else {
    final_label = "";  //Leave final label blank if invalid label
  }
}
