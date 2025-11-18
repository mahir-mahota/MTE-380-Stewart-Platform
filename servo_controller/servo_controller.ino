#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver();

// === SERVO PARAMETERS ===
#define SERVO_MIN_US  500
#define SERVO_MAX_US  2500
#define SERVO_FREQ    50.0
#define MIN_ANGLE     0
#define MAX_ANGLE     180
#define S1_OFFSET     0
#define S2_OFFSET     0
#define S3_OFFSET     3

int s1_angle = 0;
int s2_angle = 0;
int s3_angle = 0;

// === Convert microseconds to PCA9685 ticks ===
uint16_t usToTicks(uint16_t us) {
    return (uint16_t)((us * SERVO_FREQ * 4096) / 1000000);
}

// === Set servo angle on given channel ===
void setServoDeg(uint8_t ch, float deg) {
    uint16_t us = SERVO_MIN_US + (uint32_t)((deg - MIN_ANGLE) * (SERVO_MAX_US - SERVO_MIN_US) / (MAX_ANGLE - MIN_ANGLE));
    uint16_t ticks = usToTicks(us);
    pca.setPWM(ch, 0, ticks);
}

void setup() {
    Serial.begin(115200);
//    Wire.begin();
    pca.begin();
    delay(20);
    pca.setPWMFreq(SERVO_FREQ);
    delay(10);

    // Initialize servos to offsets
    setServoDeg(0, 0 + S1_OFFSET);
    setServoDeg(1, 0 + S2_OFFSET);
    setServoDeg(2, 0 + S3_OFFSET);
}

void loop() {
    // Only read input if at least 3 integers are available
    if (Serial.available() >= 3) {
        int d1 = Serial.parseInt();
        int d2 = Serial.parseInt();
        int d3 = Serial.parseInt();

        // Update internal angles safely, applying constraints
        s1_angle = constrain(d1, MIN_ANGLE, 30);
        s2_angle = constrain(d2, MIN_ANGLE, 30);
        s3_angle = constrain(d3, MIN_ANGLE, 30);

        // Send to servos, adding offsets
        setServoDeg(0, s1_angle + S1_OFFSET);
        setServoDeg(1, s2_angle + S2_OFFSET);
        setServoDeg(2, s3_angle + S3_OFFSET);

        // Print all 3 angles
        Serial.println(d1);
        Serial.println(d2);
        Serial.println(d3);
    }
}
