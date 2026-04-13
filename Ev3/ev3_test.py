from time import sleep
import ev3_dc as ev3


class Robot:
    def __init__(self, with_steering=False):
        self._ev3 = ev3.EV3(protocol=ev3.USB)
        self.motor_b = ev3.Motor(ev3.PORT_B, ev3_obj=self._ev3)
        self.motor_c = ev3.Motor(ev3.PORT_C, ev3_obj=self._ev3)
        self._has_steering = with_steering
        if with_steering:
            self.motor_a = ev3.Motor(ev3.PORT_A, ev3_obj=self._ev3)
            self.motor_a.speed = 30
        self._speed = 10

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.motor_b.stop(brake=False)
        self.motor_c.stop(brake=False)
        if self._has_steering:
            self.motor_a.stop(brake=False)

    def setSpeed(self, speed):
        self._speed = speed
        self.motor_b.speed = speed
        self.motor_c.speed = speed

    def driveStart(self, direction=1):
        self.motor_b.start_move(direction=direction)
        self.motor_c.start_move(direction=-direction)

    def driveStop(self):
        self.motor_b.stop(brake=True)
        self.motor_c.stop(brake=True)

    def setAngle(self, angle):
        if self._has_steering:
            self.motor_a.start_move_to(angle, brake=True)
        else:
            print("No steering motor connected")


with Robot(with_steering=True) as robot:
    robot.setSpeed(50)
    robot.driveStart()
    sleep(2)
    robot.driveStop()

    robot.setAngle(-90)
    sleep(20)
