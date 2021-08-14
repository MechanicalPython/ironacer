#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.
"""

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
except ImportError:
    pass

import time
import sys

import find
import strike
import telegram_bot

# todo - False negative finder: when a squirrel is missed


def main():
    camera = PiCamera()
    if len(sys.argv) == 1:
        camera.resolution = (1088, 1088)
    else:
        camera.resolution = (int(sys.argv[1]), int(sys.argv[2]))

    camera.framerate = 1
    rawCapture = PiRGBArray(camera, size=camera.resolution)

    bot = telegram_bot.TelegramBot()
    detector = find.Detector()
    claymore = strike.Claymore()

    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        image = frame.array
        s = time.time()
        squirrels = detector.darknet_detect(image)
        print(f'Time to detect: {time.time() - s}')
        if squirrels is not False:
            # Squirrel is present
            detector.save_image(squirrels, image, 'positive.jpg')
            bot.send_photo(photo_path='positive.jpg')
            claymore.detonate()

        rawCapture.truncate(0)


def test():
    image = 'test.jpg'
    bot = telegram_bot.TelegramBot()
    detector = find.Detector()
    # claymore = strike.Claymore()

    while True:
        s = time.time()
        squirrels = detector.darknet_detect(image)
        print(f'Time to detect: {time.time() - s}')
        if squirrels is not False:
            # Squirrel is present
            detector.save_image(squirrels, image, 'positive.jpg')
            # bot.send_photo(photo_path='positive.jpg')
            # claymore.detonate()
        print(f'Time for loop: {time.time() - s}')
        time.sleep(5)


if __name__ == '__main__':
    if sys.argv[1] == 'test':
        test()
    else:
        main()

