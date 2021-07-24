#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.
"""

import camera
import screen
import find
import strike
import telegram_bot


def main():
    # cam = camera.Camera('photos/')
    bot = telegram_bot.TelegramBot()
    while True:
        # Take photo
        # photo = cam.take_photo()
        photo = 'yolo/custom_data/test.jpg'
        if screen.screener(photo):
            # Squirrel is present
            squirrel_loc_stat = find.position_locater()
            bot.send_photo(photo_path=photo)
            strike.javelin(squirrel_loc_stat)
            quit()


if __name__ == '__main__':
    main()

