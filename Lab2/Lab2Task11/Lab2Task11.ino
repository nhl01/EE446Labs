#include <Arduino_HS300x.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>

//Constants
const unsigned long SAMPLE_INTERVAL = 100;
const unsigned long COOLDOWN = 1500;
const int RH_THRESHOLD = 20;
const float TEMP_THRESHOLD = 3.0;
const float MAG_MAGNITUDE_THRESHOLD = 30.0;
const int COLOR_THRESHOLD = 5;
const int CLEAR_THRESHOLD = 500;

//Baselines
float baseline_rh, baseline_temp, baseline_mag_magnitude;
int baseline_r, baseline_g, baseline_b, baseline_clear;

//Timing variables
unsigned long lastEventEndTime = 0;
unsigned long lastSampleTime = 0;

//Raw data
float x, y, z, temp, rh;
int r, g, b, clear;

//Flags and events
int humid_jump, temp_rise, mag_shift, light_or_color_change = 0;
String final_label;

void setup() {
  Serial.begin(115200);
  delay(1500);
  if (!HS300x.begin()) {
    Serial.println("Failed to initialize humidity/temperature sensor.");
    while (1)
      ;
  }

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1)
      ;
  }

  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1)
      ;
  }
  //Collect baseline values
  baseline_temp = HS300x.readTemperature();
  baseline_rh = HS300x.readHumidity();
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(x, y, z);
    baseline_mag_magnitude = calcMagMagnitude(x, y, z);
  }
  while (!APDS.colorAvailable()) {
    delay(1); //Wait until a valid color reading is ready
  }

  APDS.readColor(baseline_r, baseline_g, baseline_b, baseline_clear);

  Serial.print("BASELINE RH=");
  Serial.print(baseline_rh);
  Serial.print(",BASELINE TEMP=");
  Serial.print(baseline_temp);
  Serial.print(",BASELINE MAG=");
  Serial.print(baseline_mag_magnitude);
  Serial.print(",BASELINE R=");
  Serial.print(baseline_r);
  Serial.print(",BASELINE G=");
  Serial.print(baseline_g);
  Serial.print(",BASELINE B=");
  Serial.print(baseline_b);
  Serial.print(",BASELINE CLEAR=");
  Serial.println(baseline_clear);
}

void loop() {
  unsigned long currTime = millis();
  if (currTime - lastSampleTime >= SAMPLE_INTERVAL) { //Non-blocking "delay"
    lastSampleTime = currTime;
    temp = HS300x.readTemperature();
    rh = HS300x.readHumidity();

    if (IMU.magneticFieldAvailable()) {
      IMU.readMagneticField(x, y, z);
    }

    if (APDS.colorAvailable()) {
      APDS.readColor(r, g, b, clear);
    }
    calcFlags();
    setFinalLabel(currTime);
    Serial.print("raw,rh=");
    Serial.print(rh);
    Serial.print(",temp=");
    Serial.print(temp);
    Serial.print(",mag=<");
    Serial.print(x);
    Serial.print(",");
    Serial.print(y);
    Serial.print(",");
    Serial.print(z);
    Serial.print(">,r=");
    Serial.print(r);
    Serial.print(",g=");
    Serial.print(g);
    Serial.print(",b=");
    Serial.print(b);
    Serial.print(",clear=");
    Serial.println(clear);

    Serial.print("flags,humid_jump=");
    Serial.print(humid_jump);
    Serial.print(",temp_rise=");
    Serial.print(temp_rise);
    Serial.print(",mag_shift=");
    Serial.print(mag_shift);
    Serial.print(",light_or_color_change=");
    Serial.println(light_or_color_change);

    Serial.print("event,");
    Serial.println(final_label);
  }
}


float calcMagMagnitude(float x, float y, float z) { //Magnitude of magnetic field vector
  return sqrt(x * x + y * y + z * z);
}

void calcFlags() {
  if (abs(rh - baseline_rh > RH_THRESHOLD)) humid_jump = 1;
  else humid_jump = 0;

  if (abs(temp - baseline_temp > TEMP_THRESHOLD)) temp_rise = 1;
  else temp_rise = 0;

  if (abs(calcMagMagnitude(x, y, z) - baseline_mag_magnitude) > MAG_MAGNITUDE_THRESHOLD) mag_shift = 1;
  else mag_shift = 0;

  if (abs(r - baseline_r) > COLOR_THRESHOLD || abs(g - baseline_g) > COLOR_THRESHOLD || abs(b - baseline_b) > COLOR_THRESHOLD || abs(clear - baseline_clear) > CLEAR_THRESHOLD) light_or_color_change = 1;
  else light_or_color_change = 0;
}

void setFinalLabel(unsigned long currTime) {
  String candidate = "";
  if(!humid_jump && !temp_rise && !mag_shift && !light_or_color_change) {
    candidate = "BASELINE_NORMAL";
  } else if (humid_jump || temp_rise) {
    candidate = "BREATH_OR_WARM_AIR_EVENT";
  } else if (mag_shift) {
    candidate = "MAGNETIC_DISTURBANCE_EVENT";
  } else if (light_or_color_change) {
    candidate = "LIGHT_OR_COLOR_CHANGE_EVENT";
  }

  if(candidate == "BASELINE_NORMAL") {
    String prevLabel = final_label;
    final_label = candidate;
    if(prevLabel != "BASELINE_NORMAL") {
    lastEventEndTime = currTime; //Cooldown starts when returning to baseline
    }
  } else if (currTime - lastEventEndTime >= COOLDOWN) {
    final_label = candidate; //Only update event away from baseline if cooldown has passed
  }
}
