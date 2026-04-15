import time
import math
import sys
import os

# Allow importing hardware.py from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hardware import Robot


# -------- CONFIG (PUT YOUR TUNED VALUES HERE) --------
robot = Robot()

# Wave settings
FREQUENCY = 0.5      # Hz (cycles per second)
DT = 0.02            # loop timestep


def main():
    print("Starting sine drive test (motor + steering)...")
    print("Press CTRL+C to stop.")

    t = 0.0

    try:
        while True:
            # Smooth sine wave
            motor = math.sin(2 * math.pi * FREQUENCY * t)
            steering = math.sin(2 * math.pi * FREQUENCY * t + math.pi / 2)

            robot.set_throttle(motor)
            robot.set_steering(steering)

            print(f"\rThrottle: {motor:+.2f} | Steering: {steering:+.2f}", end="")

            t += DT
            time.sleep(DT)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Safe shutdown
        try:
            robot.set_throttle(0)
            robot.set_steering(0)
            robot.stop()
        except Exception as e:
            print("Cleanup error:", e)


if __name__ == "__main__":
    main()