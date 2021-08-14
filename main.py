#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.
"""

from picamera import PiCamera
from picamera.array import PiRGBArray
import time

import find
import strike
import telegram_bot
import camera

# todo - False negative finder: when a squirrel is missed


def main():
    camera = PiCamera()
    camera.resolution = (1088, 1088)
    camera.framerate = 1
    rawCapture = PiRGBArray(camera, size=camera.resolution)

    # bot = telegram_bot.TelegramBot()
    detector = find.Detector()

    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        image = frame.array
        s = time.time()
        squirrels = detector.darknet_detect(image)
        print(f'Time to detect: {time.time() - s}')
        if squirrels is not False:
            # Squirrel is present
            detector.save_image(squirrels, image, 'positive.jpg')
            # bot.send_photo(photo_path='positive.jpg')
            strike.claymore()

        rawCapture.truncate(0)


if __name__ == '__main__':
    main()

