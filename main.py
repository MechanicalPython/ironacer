#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.

Workflow
Camera image -> yolov5 -> if squirrel -> fire mechanism and send photo, else do nothing.

"""

import datetime
import subprocess
import time

from tenacity import retry, wait_fixed, retry_if_exception_type

import find
import telegram_bot
import sys

# todo - False negative finder: when a squirrel is missed
# todo - telegram to send photos to chat on request.

# todo - auto get sunrise and sunset.


@retry(wait=wait_fixed(60), retry=retry_if_exception_type(AssertionError))
def main():
    print('Starting Stream')
    subprocess.Popen(["ssh", "pi@ironacer.local", "python3 stream.py"],
                     stdin=None, stdout=None, stderr=None, close_fds=True)
    time.sleep(5)

    print('Activating IRONACER')
    bot = telegram_bot.TelegramBot()
    # claymore = strike.Claymore()

    d = find.StreamDetector(weights='best.pt')
    for i in d.stream():
        isSquirrel, coords, confidence, vid_path = i
        print(isSquirrel, coords, confidence, vid_path)

        if isSquirrel:  # Squirrel is present
            # claymore.detonate()
            pass
        if vid_path is not False:
            bot.send_video(vid_path=vid_path)

        now = datetime.datetime.now()
        sunset = datetime.datetime(year=now.year, month=now.month, day=now.day, hour=16, minute=37)
        print(now, sunset)
        if now > sunset:
            subprocess.Popen(["ssh", "pi@ironacer.local", "pkill -f stream.py"],
                             stdin=None, stdout=None, stderr=None, close_fds=True)
            return None


if __name__ == '__main__':
    main()
