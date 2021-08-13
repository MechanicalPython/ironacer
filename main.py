#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.
"""

from picamera import PiCamera
from picamera.array import PiRGBArray

import find
import strike
import telegram_bot


# todo - False negative finder: when a squirrel is missed


def main():
    camera = PiCamera()
    camera.resolution = (2592, 1944)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(2592, 1944))

    bot = telegram_bot.TelegramBot()
    detector = find.Detector()

    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        image = frame.array
        squirrels = detector.darknet_detect(image)
        if squirrels is not False:
            # Squirrel is present
            detector.save_image(squirrels, image, 'positive.jpg')
            bot.send_photo(photo_path='positive.jpg')
            strike.claymore()

        rawCapture.truncate(0)



    while True:
        # Take photo
        photo = cam.stream_photo()




if __name__ == '__main__':
    main()

