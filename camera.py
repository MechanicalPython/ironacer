#! /usr/local/bin/python3.7

try:
    from picamera import PiCamera
except ImportError:
    pass
from time import sleep
from datetime import datetime
import os


class Camera:
    """
    Basic camera module that, when initiated, will open the camera ready for photos to be taken.
    Init Camera when starting the program to avoid time taken to adjust for brightness, etc.

    """
    def __init__(self, output_dir):
        if not output_dir.endswith('/'):
            output_dir = f'{output_dir}/'
        self.output_dir = os.path.expanduser(output_dir)

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.camera = PiCamera()
        self.camera.rotation = 180
        self.camera.resolution = (2592, 1944)  # Max resolution requires 15 fps.
        self.camera.framerate = 15
        self.camera.start_preview()
        sleep(3)  # Sleep to allow camera to adjust to the light

    def __exit__(self):
        self.camera.stop_preview()

    def take_photo(self):
        """
        Takes single photo called YYYYMMdd-HHMMSS-f.jpg to given directory.
        :return:
        """
        now = datetime.now().strftime('%Y%m%d-%H%M%S,%f')
        self.camera.capture(f'{self.output_dir}{now}.jpg')
        return f'{self.output_dir}{now}.jpg'

    def take_burst(self, number):
        """
        Takes a burst of photos as fast as possible.
        """
        for i in range(number):
            photo_path = self.take_photo()
        return photo_path
