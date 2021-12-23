#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.
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
    # take initial photo and send it to the telegram chat.
    cam.take_photo('init.jpg')
    bot.send_photo('init.jpg')

    detector = find.Detector()
    claymore = strike.Claymore()

    for image in cam.stream_photo():
        s = time.time()
        squirrels = detector.detect(image)
        print(f'Time to detect: {time.time() - s}')
        if len(squirrels) > 0:
            # Squirrel is present

            detector.save_image(squirrels, image, 'positive.jpg')
            bot.send_photo(photo_path='positive.jpg')
            claymore.detonate()


def test():
    image = 'test.jpg'
    bot = telegram_bot.TelegramBot()
    detector = find.Detector()
    # claymore = strike.Claymore()

    while True:
        s = time.time()
        squirrels = detector.detect(image)
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

