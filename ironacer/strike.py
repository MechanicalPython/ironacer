"""
Strike.py - employ the anti-squirrel measures.
"""
import time
import RPi.GPIO as GPIO


class ServoController:
    """
    Can rotate 0-180 degrees, or continuously turn clockwise/anti-clockwise (1 turn in 0.83 seconds)
    Control servo motors to aim the gun.
    Servo motor has a period of 50Hz (20ms).
    The position is tied to the pulse width. So a 1.5ms pulse over a 20ms duration is position 0.

    """
    def __init__(self, control_pin=17):
        self.pin = control_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.p = GPIO.PWM(self.pin, 50)

    def rotate(self, degrees):
        """Rotate the servo arm a number of degrees.
        Can only rotate through 180 degrees, from 3 (0) anti-clockwise to 13 (180). 8 is 90"""
        if degrees > 180:
            raise Exception('Input cannot be above 180')
        turn_amount = (degrees / 180) * 10  # Gives 0-10, being 0 to 180 degree turn.
        self.p.ChangeDutyCycle(turn_amount + 3)
        time.sleep(0.5)
        self.turn_off()

    def reset(self):
        """Reset the arm too forwards."""
        self.p.ChangeDutyCycle(2.5)
        time.sleep(0.5)
        self.turn_off()

    def turn_off(self):
        """Essentially 'turns off' the arm, so it stays where it is"""
        self.p.ChangeDutyCycle(0)

    def clockwise(self):
        """Continuous clockwise rotation """
        self.p.ChangeDutyCycle(50)

    def anti_clockwise(self):
        """Continuous anti-clockwise rotation """
        self.p.ChangeDutyCycle(15)


class Relay:
    relay_pins = {"R1": 31, "R2": 33, "R3": 35, "R4": 37}

    def __init__(self, pins):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        self.pin = self.relay_pins[pins]
        self.pins = pins
        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, GPIO.LOW)

    def on(self):
        GPIO.output(self.pin, GPIO.HIGH)

    def off(self):
        GPIO.output(self.pin, GPIO.LOW)


class Claymore:
    """Initiates an AOE blast"""
    def __init__(self):
        self.firing_pin = Relay('R1')
        self.is_on = False

    def start(self):
        self.firing_pin.on()
        self.is_on = True

    def stop(self):
        if self.is_on:
            self.firing_pin.off()
            self.is_on = False

    def timed_exposure(self, duration):
        self.start()
        time.sleep(duration)
        self.stop()



