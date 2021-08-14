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
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=camera.resolution)

    # bot = telegram_bot.TelegramBot()
    # detector = find.Detector()
    s = time.time()
    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        image = frame.array
        print(f'{time.time() - s}')

        # squirrels = detector.darknet_detect(image)
        # print(squirrels)
        # if squirrels is not False:
        #     # Squirrel is present
        #     detector.save_image(squirrels, image, 'positive.jpg')
        #     # bot.send_photo(photo_path='positive.jpg')
        #     strike.claymore()

        rawCapture.truncate(0)


if __name__ == '__main__':
    main()

