"""
Strike.py - employ the anti-squirrel measures.
"""


import RPi.GPIO as GPIO
from time import sleep


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

    def detonate(self):
        self.firing_pin.on()
        sleep(2)
        self.firing_pin.off()
