"""
Strike.py - employ the anti-squirrel measures.
"""


import RPi.GPIO as GPIO
from time import sleep
import threading


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
    def __init__(self, duration=2):
        self.firing_pin = Relay('R1')
        self.duration = duration

    def detonate(self):
        self.firing_pin.on()
        sleep(self.duration)
        self.firing_pin.off()


def threaded_strike(duration=2):
    claymore = Claymore(duration)
    threading.Thread(target=claymore.detonate, daemon=True).start()
