/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Adapted for ULN2003AN dual stepper motor cart control.
 * Motors are driven via AccelStepper in HALF4WIRE mode (28BYJ-48 compatible).
 * Commands: "forward", "backward", "left", "right", "stop"
 *
 * Continuous motion model:
 *   - A direction command sets the active state and motors run continuously.
 *   - Motors keep moving during the mic recording/inference window.
 *   - Only a new direction command or "stop" changes motor behaviour.
 *   - Low-confidence results and "unknown" are silently ignored.
 *
 * LCD (16x2 I2C, hardware I2C on A4=SDA, A5=SCL):
 *   Line 1 — "Current:<direction>"  reflects activeState
 *   Line 2 — "Heard:<label>"        reflects last classifier decision
 *   LCD is refreshed on every loop iteration so it always reflects
 *   the live state, not just on change events.
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK 0

/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <VoiceControlledCart_inferencing.h>
#include <AccelStepper.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ============================================================
// ACTIVE COMMAND STATE
// — declared first so CartState is in scope for all functions
// ============================================================
enum CartState {
    STATE_STOPPED,
    STATE_FORWARD,
    STATE_BACKWARD,
    STATE_LEFT,
    STATE_RIGHT
};

static CartState activeState  = STATE_STOPPED;

// Tracks the last heard label for LCD line 2
static String lastHeardLabel  = "---";

// ============================================================
// LCD CONFIGURATION
// — hardware I2C on A4 (SDA) and A5 (SCL)
// The Nano 33 BLE Sense uses MbedI2C which does NOT support
// pin remapping — A4/A5 are the only valid I2C pins.
// ============================================================
// Default I2C address for PCF8574-based backpacks is 0x27.
// If the display is blank after boot, change this to 0x3F.
#define LCD_I2C_ADDR  0x27
#define LCD_COLS      16
#define LCD_ROWS      2


LiquidCrystal_I2C lcd(LCD_I2C_ADDR, LCD_COLS, LCD_ROWS);

/**
 * @brief Convert a CartState to its display string.
 */
String stateToString(CartState s) {
    switch (s) {
        case STATE_FORWARD:  return "Forward";
        case STATE_BACKWARD: return "Backward";
        case STATE_LEFT:     return "Left";
        case STATE_RIGHT:    return "Right";
        case STATE_STOPPED:
        default:             return "Stopped";
    }
}

/**
 * @brief Overwrite both LCD lines with current state and last
 *        heard label. Called every loop iteration so the display
 *        always reflects live values without any change-detection
 *        logic that could cause it to fall out of sync.
 */
void refreshLCD() {
    // Line 1 — current moving direction
    lcd.setCursor(0, 0);
    String line1 = "Current:" + stateToString(activeState);
    // Pad to 16 chars to overwrite any leftover characters
    while (line1.length() < LCD_COLS) line1 += ' ';
    lcd.print(line1);

    // Line 2 — last heard command
    lcd.setCursor(0, 1);
    String line2 = "Heard:" + lastHeardLabel;
    while (line2.length() < LCD_COLS) line2 += ' ';
    lcd.print(line2);
}

// ============================================================
// MOTOR PIN DEFINITIONS
// ============================================================
// Left motor  — ULN2003AN board IN1..IN4
#define L_IN1 2
#define L_IN2 3
#define L_IN3 4
#define L_IN4 5

// Right motor — ULN2003AN board IN1..IN4
#define R_IN1 6
#define R_IN2 7
#define R_IN3 8
#define R_IN4 9

// AccelStepper objects in HALF4WIRE mode (8-step half-step sequence)
// Pin order must be IN1, IN3, IN2, IN4 for correct 28BYJ-48 sequencing
AccelStepper leftMotor(AccelStepper::HALF4WIRE,  L_IN1, L_IN3, L_IN2, L_IN4);
AccelStepper rightMotor(AccelStepper::HALF4WIRE, R_IN1, R_IN3, R_IN2, R_IN4);

// ============================================================
// MOTOR CONFIGURATION
// ============================================================
#define STEPS_PER_REV   2048

#define DRIVE_STEPS     4096
#define TURN_STEPS      512

#define MAX_SPEED       1200.0
#define ACCELERATION    600.0

// Confidence threshold — predictions below this are ignored
#define CONFIDENCE_THRESHOLD 0.90

// ============================================================
// AUDIO INFERENCE STRUCTS
// ============================================================
typedef struct {
    int16_t *buffer;
    uint8_t  buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false;

// ============================================================
// MOTOR HELPERS
// ============================================================

void motorsInit() {
    leftMotor.setMaxSpeed(MAX_SPEED);
    leftMotor.setAcceleration(ACCELERATION);
    rightMotor.setMaxSpeed(MAX_SPEED);
    rightMotor.setAcceleration(ACCELERATION);
}

/**
 * @brief Queue another chunk of steps in the current direction
 *        if the motor's step buffer is running low, then advance
 *        each motor by one step toward its target. Must be called
 *        as often as possible to keep motion smooth and continuous.
 */
void keepMoving() {
    switch (activeState) {

        case STATE_FORWARD:
            if (leftMotor.distanceToGo()  == 0) leftMotor.move(DRIVE_STEPS);
            if (rightMotor.distanceToGo() == 0) rightMotor.move(-DRIVE_STEPS);
            break;

        case STATE_BACKWARD:
            if (leftMotor.distanceToGo()  == 0) leftMotor.move(-DRIVE_STEPS);
            if (rightMotor.distanceToGo() == 0) rightMotor.move(DRIVE_STEPS);
            break;

        case STATE_LEFT:
            if (leftMotor.distanceToGo()  == 0) leftMotor.move(-TURN_STEPS);
            if (rightMotor.distanceToGo() == 0) rightMotor.move(-TURN_STEPS);
            break;

        case STATE_RIGHT:
            if (leftMotor.distanceToGo()  == 0) leftMotor.move(TURN_STEPS);
            if (rightMotor.distanceToGo() == 0) rightMotor.move(TURN_STEPS);
            break;

        case STATE_STOPPED:
        default:
            break;
    }

    // Advance both motors one step toward their targets (non-blocking)
    leftMotor.run();
    rightMotor.run();
}

/**
 * @brief Transition to a new CartState.
 *        Enables outputs and seeds the first batch of steps.
 */
void setActiveState(CartState newState) {
    activeState = newState;

    if (newState == STATE_STOPPED) {
        leftMotor.stop();
        rightMotor.stop();
        leftMotor.disableOutputs();
        rightMotor.disableOutputs();
        ei_printf("CMD: stop — motors halted and de-energized.\n");
    } else {
        leftMotor.enableOutputs();
        rightMotor.enableOutputs();
        // Reset step buffer so keepMoving() seeds fresh steps immediately
        leftMotor.move(0);
        rightMotor.move(0);
    }
}

// ============================================================
// COMMAND DISPATCH
// ============================================================

/**
 * @brief Evaluate classifier result, update activeState, and
 *        update lastHeardLabel for the LCD.
 *
 *  - Below confidence threshold  → lastHeardLabel = "unknown", motors unchanged
 *  - "unknown" label             → lastHeardLabel = "unknown", motors unchanged
 *  - direction / stop command    → update both activeState and lastHeardLabel
 */
void dispatchCommand(ei_impulse_result_t &result) {
    float      max_val   = 0.0;
    const char *max_label = "";

    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (result.classification[ix].value > max_val) {
            max_val   = result.classification[ix].value;
            max_label = result.classification[ix].label;
        }
    }

    ei_printf("Best prediction: %s (%.3f)\n", max_label, max_val);

    // Gate 1 — confidence too low, keep current state
    if (max_val < CONFIDENCE_THRESHOLD) {
        ei_printf("Confidence too low — maintaining current state.\n");
        lastHeardLabel = "unknown";
        return;
    }

    // Gate 2 — unknown/noise, keep current state
    if (strcmp(max_label, "unknown") == 0) {
        ei_printf("Unknown command — maintaining current state.\n");
        lastHeardLabel = "unknown";
        return;
    }

    // Dispatch recognised commands
    if (strcmp(max_label, "forward") == 0) {
        ei_printf("CMD: forward\n");
        lastHeardLabel = "Forward";
        setActiveState(STATE_FORWARD);
    }
    else if (strcmp(max_label, "backward") == 0) {
        ei_printf("CMD: backward\n");
        lastHeardLabel = "Backward";
        setActiveState(STATE_BACKWARD);
    }
    else if (strcmp(max_label, "left") == 0) {
        ei_printf("CMD: left\n");
        lastHeardLabel = "Left";
        setActiveState(STATE_LEFT);
    }
    else if (strcmp(max_label, "right") == 0) {
        ei_printf("CMD: right\n");
        lastHeardLabel = "Right";
        setActiveState(STATE_RIGHT);
    }
    else if (strcmp(max_label, "stop") == 0) {
        ei_printf("CMD: stop\n");
        lastHeardLabel = "Stopped";
        setActiveState(STATE_STOPPED);
    }
    else {
        ei_printf("Unhandled label '%s' — maintaining current state.\n", max_label);
        lastHeardLabel = "unknown";
    }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Voice-Controlled Cart — Edge Impulse + AccelStepper + LCD");

    // --- LCD init on hardware I2C: A4=SDA, A5=SCL ---
    Wire.begin(); // hardware I2C — A4=SDA, A5=SCL (fixed on Nano 33 BLE Sense)
    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("Current:Stopped ");
    lcd.setCursor(0, 1);
    lcd.print("Heard:---       ");

    // --- Motor init ---
    motorsInit();
    setActiveState(STATE_STOPPED);

    // --- EI model info ---
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n",   (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n",        EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n",
        sizeof(ei_classifier_inferencing_categories) /
        sizeof(ei_classifier_inferencing_categories[0]));

    if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d)\r\n",
                  EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        lcd.setCursor(0, 0);
        lcd.print("ERR: mic alloc  ");
        return;
    }
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {

    // Refresh LCD at the top of every loop so it always shows live state
    refreshLCD();

    ei_printf("\n[MIC] Recording has begun — say your command (%d ms window).\n",
              EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);

    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio\n");
        return;
    }

    ei_printf("[MIC] Recording complete — running inference.\n");

    // Keep motors moving while inference runs
    keepMoving();

    // Refresh LCD mid-loop so display stays current during inference latency
    refreshLCD();

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data     = &microphone_audio_signal_get_data;

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    // Print all class scores to Serial
    ei_printf("Predictions (DSP: %d ms, Classification: %d ms):\n",
              result.timing.dsp, result.timing.classification);
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n",
                  result.classification[ix].label,
                  result.classification[ix].value);
    }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

    // Dispatch — updates activeState and lastHeardLabel
    dispatchCommand(result);

    // Refresh LCD immediately after dispatch so new state shows at once
    refreshLCD();

    // Seed motors for the new (or unchanged) state
    keepMoving();
}

// ============================================================
// PDM / MICROPHONE FUNCTIONS
// ============================================================

static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (inference.buf_ready == 0) {
        for (int i = 0; i < bytesRead >> 1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];
            if (inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}

static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (inference.buffer == NULL) return false;

    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    PDM.onReceive(&pdm_data_ready_inference_callback);
    PDM.setBufferSize(4096);

    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        microphone_inference_end();
        return false;
    }

    PDM.setGain(127);
    return true;
}

/**
 * @brief Block until the mic buffer is full, pumping motors and
 *        refreshing the LCD on every iteration so nothing stalls.
 */
static bool microphone_inference_record(void) {
    inference.buf_ready = 0;
    inference.buf_count = 0;
    while (inference.buf_ready == 0) {
        keepMoving();
        refreshLCD();
    }
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

static void microphone_inference_end(void) {
    PDM.end();
    free(inference.buffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
