#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver();

#define SERVO_MIN_US  500
#define SERVO_MAX_US  2500
#define SERVO_FREQ    50.0
#define MIN_ANGLE     0
#define MAX_ANGLE     50
#define S1_OFFSET     0
#define S2_OFFSET     0
#define S3_OFFSET     3

int s1_angle = S1_OFFSET;
int s2_angle = S2_OFFSET;
int s3_angle = S3_OFFSET;

uint16_t usToTicks(uint16_t us) {
    return (uint16_t)((us * SERVO_FREQ * 4096.0) / 1000000.0);
}

void setServoDeg(uint8_t ch, float deg) {
    deg = constrain(deg, MIN_ANGLE, 30); // safety check
    uint16_t us = SERVO_MIN_US + (uint32_t)((deg - MIN_ANGLE) * (SERVO_MAX_US - SERVO_MIN_US) / (MAX_ANGLE - MIN_ANGLE));
    uint16_t ticks = usToTicks(us);
    pca.setPWM(++ch, 0, ticks);
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    pca.begin();
    delay(20);
    pca.setPWMFreq(SERVO_FREQ);
    delay(10);

    // Initialize servos to offsets
    setServoDeg(1, s1_angle);
    setServoDeg(2, s2_angle);
    setServoDeg(3, s3_angle);
}

void loop() {
    // Only read input if at least 3 integers are available
    if (Serial.available() >= 3) {
        int d1 = Serial.parseInt();
        int d2 = Serial.parseInt();
        int d3 = Serial.parseInt();

        s1_angle = constrain(s1_angle + d1, MIN_ANGLE, MAX_ANGLE);
        s2_angle = constrain(s2_angle + d2, MIN_ANGLE, MAX_ANGLE);
        s3_angle = constrain(s3_angle + d3, MIN_ANGLE, MAX_ANGLE);

        setServoDeg(0, s1_angle);
        setServoDeg(1, s2_angle);
        setServoDeg(2, s3_angle);
    }
}