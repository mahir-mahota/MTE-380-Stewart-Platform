#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver();

#define SERVO_MIN_US  500;
#define SERVO_MAX_US  2500;
#define SERVO_FREQ    50.0;
#define MIN_ANGLE     0;
#define MAX_ANGLE     50;
#define S1_OFFSET     0; 
#define S2_OFFSET     0; 
#define S3_OFFSET     3;

char buffer[100];
int s1_angle = 0;
int s2_angle = 0;
int s3_angle = 0;

uint16_t usToTicks(uint16_t us) {
  // 12-bit (4096 steps) over period T = 1/freq
  // ticks = us * freq * 4096 / 1e6
  return (uint16_t)((us * SERVO_FREQ * 4096.0) / 1000000.0);
}

void setServoDeg(uint8_t ch, float deg) {
  deg = constrain(deg, MIN_ANGLE, MAX_ANGLE);
  uint16_t us = SERVO_MIN_US + (uint32_t)(deg * (SERVO_MAX_US - SERVO_MIN_US) / 180.0);
  uint16_t ticks = usToTicks(us);
  pca.setPWM(--ch, 0, ticks);
}

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  Wire.begin();
  pca.begin();
  pca.setPWMFreq(SERVO_FREQ);
  delay(10);
  
  setServoDeg(1, 0 + SERVO_0_OFFSET);
  setServoDeg(2, 0 + SERVO_1_OFFSET);
  setServoDeg(3, 0 + SERVO_2_OFFSET);
}

void loop() {
  if (Serial.available() > 0) {
    s1_angle += Serial.parseInt();
    s2_angle += Serial.parseInt();
    s3_angle += Serial.parseInt();

    sprintf(buffer, "S1: %d\r\nS2: %d\r\nS3: %d\r\n", s1_angle, s2_angle, s3_angle);
    Serial.print(buffer);\

    setServoDeg(0, s1_angle);
    setServoDeg(1, s2_angle);
    setServoDeg(2, s3_angle);
  }
}