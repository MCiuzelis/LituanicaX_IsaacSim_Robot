import RPi.GPIO as GPIO


class Robot:
    def __init__(
        self,
        pwm_motor_pin=18,
        dir_motor_pin=15,
        servo_pin=12,
        pwm_freq=1000,
        servo_freq=50,
        servo_min_us=500,
        servo_max_us=2500,
        servo_center_us=1500,
        servo_offset_us=-130,
        servo_amplitude=0.25,
        servo_direction=1,
        motor_direction=-1,
        max_throttle=0.2
    ):
        # Store config
        self.pwm_motor_pin = pwm_motor_pin
        self.dir_motor_pin = dir_motor_pin
        self.servo_pin = servo_pin

        self.pwm_freq = pwm_freq
        self.servo_freq = servo_freq

        self.servo_min_us = servo_min_us
        self.servo_max_us = servo_max_us
        self.servo_center_us = servo_center_us
        self.servo_offset_us = servo_offset_us

        self.servo_amplitude = servo_amplitude
        self.servo_direction = servo_direction
        self.motor_direction = motor_direction

        # ✅ Clamp and store max throttle
        self.max_throttle = max(0.0, min(1.0, max_throttle))

        # GPIO setup
        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.dir_motor_pin, GPIO.OUT)
        GPIO.setup(self.pwm_motor_pin, GPIO.OUT)
        GPIO.setup(self.servo_pin, GPIO.OUT)

        # Create PWM instances
        self.motor_pwm = GPIO.PWM(self.pwm_motor_pin, self.pwm_freq)
        self.servo_pwm = GPIO.PWM(self.servo_pin, self.servo_freq)

        # Start PWM (0 duty initially)
        self.motor_pwm.start(0)
        self.servo_pwm.start(0)

    # ---------------- MOTOR ----------------
    def set_throttle(self, value):
        """
        value: -1.0 to 1.0
        """
        value = max(-1.0, min(1.0, value))

        # ✅ Apply max throttle limit
        value *= self.max_throttle

        # Apply direction inversion
        value *= self.motor_direction

        direction = GPIO.HIGH if value >= 0 else GPIO.LOW
        speed = abs(value)

        GPIO.output(self.dir_motor_pin, direction)

        # Duty cycle is 0–100
        self.motor_pwm.ChangeDutyCycle(speed * 100.0)

    # ---------------- SERVO ----------------
    def set_steering(self, value):
        """
        value: -1.0 to 1.0
        0 = center
        """
        value = max(-1.0, min(1.0, value))

        # Apply direction + amplitude
        value = value * self.servo_direction * self.servo_amplitude

        # Convert to pulse width
        span = (self.servo_max_us - self.servo_min_us) / 2
        pulse = (
            self.servo_center_us
            + self.servo_offset_us
            + (value * span)
        )

        # Clamp
        pulse = max(self.servo_min_us, min(self.servo_max_us, pulse))

        # Convert µs → duty cycle (20ms period)
        duty = (pulse / 20000.0) * 100.0

        self.servo_pwm.ChangeDutyCycle(duty)

    # ---------------- CLEANUP ----------------
    def stop(self):
        try:
            self.motor_pwm.ChangeDutyCycle(0)
            self.servo_pwm.ChangeDutyCycle(0)

            self.motor_pwm.stop()
            self.servo_pwm.stop()
        finally:
            GPIO.cleanup()