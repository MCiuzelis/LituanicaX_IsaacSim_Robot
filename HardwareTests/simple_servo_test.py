import RPi.GPIO as GPIO
import time
import math

PIN = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

# 50 Hz servo PWM
pwm = GPIO.PWM(PIN, 50)
pwm.start(0)

def angle_to_duty(angle):
    # Map 0–180° → ~2.5–12.5% duty cycle
    return 2.5 + (angle / 180.0) * 10.0

try:
    t = 0
    while True:
        # sine wave from 0–180 degrees
        angle = (math.sin(t) + 1) * 90  # 0..180
        duty = angle_to_duty(angle)

        pwm.ChangeDutyCycle(duty)
        time.sleep(0.02)

        t += 0.05

except KeyboardInterrupt:
    pass

pwm.stop()
GPIO.cleanup()