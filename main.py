#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.

Workflow
Camera image -> yolov5 -> if squirrel -> fire mechanism and send photo, else do nothing.

"""

import time
import sys

import find
import strike
import telegram_bot
import camera

# todo - False negative finder: when a squirrel is missed
# todo - telegram to send photos to chat on request.


def main():
    cam = camera.Camera()
    if len(sys.argv) == 1:
        cam.resolution = (1088, 1088)
    else:
        cam.resolution = (int(sys.argv[1]), int(sys.argv[2]))

    bot = telegram_bot.TelegramBot()
    detector = find.Detector()
    claymore = strike.Claymore()

    for image in cam.stream_photo():
        s = time.time()
        squirrels, coords, picture = detector.detect(image)
        print(f'Time to detect: {time.time() - s}')

        if squirrels:  # Squirrel is present
            bot.send_photo(photo_path=picture)
            claymore.detonate()


if __name__ == '__main__':
    main()

