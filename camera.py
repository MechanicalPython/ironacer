#! /usr/local/bin/python3.9

from picamera import PiCamera
from time import sleep
from datetime import datetime
from os.path import expanduser


class Camera:
    def __init__(self):
        self.home_dir = f'{expanduser("~")}/ironacer_photos/'
        self.camera = PiCamera()
        self.camera.rotation = 180
        self.camera.resolution = (2592, 1944)  # Max resolution requires 15 fps.
        self.camera.framerate = 15
        self.camera.start_preview()
        sleep(3)  # Sleep to allow camera to adjust to the light

    def __exit__(self):
        self.camera.stop_preview()

    def take_photo(self):
        now = datetime.now().strftime('%Y%m%d-%H%M%S,%f')
        self.camera.capture(f'{self.home_dir}{now}.jpg')
        return f'{self.home_dir}{now}.jpg'

    def take_burst(self, number):
        for i in range(number):
            photo_path = self.take_photo()
            sleep(0.5)
        return photo_path


if __name__ == '__main__':
    camera = Camera()
    for i in range(10):
        camera.take_photo()
        sleep(1)




