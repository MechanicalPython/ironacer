#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.
"""

import camera
import find
import strike
import telegram_bot


# todo - False negative finder: when a squirrel is missed


def main():
    cam = camera.Camera()
    bot = telegram_bot.TelegramBot()
    detector = find.Detector()
    while True:
        # Take photo
        photo = cam.take_photo()
        # Select random from pre-list for testing.

        squirrels = detector.darknet_detect(photo)
        if squirrels is not False:
            # Squirrel is present
            detector.save_image(squirrels, photo, 'positive.jpg')
            bot.send_photo(photo_path='positive.jpg')
            strike.claymore()


if __name__ == '__main__':
    main()

