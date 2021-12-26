#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.

Workflow
Camera image -> yolov5 -> if squirrel -> fire mechanism and send photo, else do nothing.

"""

import find
import strike
import telegram_bot

from tenacity import retry, wait_fixed, retry_if_exception_type

# todo - False negative finder: when a squirrel is missed
# todo - telegram to send photos to chat on request.


@retry(wait=wait_fixed(60), retry=retry_if_exception_type(AssertionError))
def main():
    print('Activating IRONACER')
    bot = telegram_bot.TelegramBot()
    # claymore = strike.Claymore()

    for i in find.detect_stream(source='http://ironacer.local:8000/stream.mjpg'):
        isSquirrel, coords, confidence, vid_path = i

        if isSquirrel:  # Squirrel is present
            if vid_path is not False:
                bot.send_video(vid_path=vid_path)
            # claymore.detonate()


if __name__ == '__main__':
    main()

