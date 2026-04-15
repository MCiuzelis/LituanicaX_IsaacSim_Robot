import time
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hardware import Robot

robot = Robot(
    motor_direction=-1
)

try:
    print("Running sine wave test...")

    t = 0.0
    dt = 0.05

    while True:
        motor_val = math.sin(t)
        servo_val = math.sin(t + math.pi / 2)

        robot.set_throttle(motor_val)
        print(f"Motor: {motor_val:.2f}")

        t += dt
        time.sleep(dt)

except KeyboardInterrupt:
    print("\nInterrupt received — stopping motor immediately!")

    # 🚨 immediate hard stop (do not wait for finally)
    try:
        robot.set_throttle(0)
        robot.set_steering(0)
        robot.stop()
    except Exception as e:
        print("Error during emergency stop:", e)

finally:
    # backup cleanup (safe even if already stopped)
    try:
        robot.stop()
    except:
        pass