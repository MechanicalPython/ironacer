#! /usr/local/bin/python3.7

"""
Strike.py - employ the anti-squirrel measures.
"""

# todo - convert camera distance and angle to target to gun distance and angle.
# todo - locstat to trajectory required

import RPi.GPIO as GPIO
from time import sleep


GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)


class Relay:
    relay_pins = {"R1": 31, "R2": 33, "R3": 35, "R4": 37}

    def __init__(self, pins):
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

    def detonate(self):
        self.firing_pin.on()
        sleep(5)
        self.firing_pin.off()


def javelin(loc_stat):
    """
    :param: loc_stat - the location of the squirrel.
    Launch stike against squirrel
    :return:
    """
    pass
