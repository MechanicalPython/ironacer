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

# todo - 1. pi-running motion detector to gather images of squirrels.
#  telegram to send photos to chat on request.
#  auto get sunrise and sunset.
#  test speed of new weights.
#  record the frame rate while running inference on it.


@retry(wait=wait_fixed(60), retry=retry_if_exception_type(AssertionError))
def main():
    try:
        print('Starting Stream')
        subprocess.Popen(["ssh", "pi@ironacer.local", "python3 stream.py"],
                         stdin=None, stdout=None, stderr=None, close_fds=True)
        time.sleep(5)
        print('Activating IRONACER')
        bot = telegram_bot.TelegramBot()
        # claymore = strike.Claymore()

        d = find.StreamDetector(weights='yolov5n6_best.pt', imgsz=(1280, 1280))
        for path, im, im0s, vid_cap, s in d.stream():
            d.motion_detector(im0s[0])  # Saves motion detected images.
            isSquirrel, coords, confidence, vid_path = d.inference(path, im, im0s, vid_cap, s)  # Runs yolov5 inference.

            if isSquirrel:  # Squirrel is present
                # claymore.detonate()
                pass
            if vid_path is not False:
                bot.send_video(vid_path=vid_path)
            now = datetime.datetime.now()
            sunset = datetime.datetime(year=now.year, month=now.month, day=now.day, hour=16, minute=37)
            if now > sunset:
                return None
    finally:
        subprocess.Popen(["ssh", "pi@ironacer.local", "pkill -f stream.py"],
                         stdin=None, stdout=None, stderr=None, close_fds=True)


if __name__ == '__main__':
    main()
