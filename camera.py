#! /usr/local/bin/python3.7

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
except ImportError:
    pass
from time import sleep


class Camera:
    """
    Basic camera module that, when initiated, will open the camera ready for photos to be taken.
    Init Camera when starting the program to avoid time taken to adjust for brightness, etc.

    Use camera stream to avoid writing to jpg constantly.
    """
    def __init__(self):
        self.camera = PiCamera()
        self.raw_caputre = PiRGBArray(self.camera)
        self.camera.rotation = 180
        self.camera.resolution = (2592, 1952)  # Max resolution requires 15 fps.
        self.camera.framerate = 30
        self.camera.start_preview()
        sleep(3)  # Sleep to allow camera to adjust to the light

    def __exit__(self):
        self.camera.stop_preview()

    def take_photo(self):
        self.camera.capture('')

    def stream_photo(self):
        """
        Takes single photo as image array to be passed to opencv.
        :return:
        """
        self.camera.capture(self.raw_caputre, format='bgr')
        image = self.raw_caputre.array
        return image
