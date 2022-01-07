#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.

Workflow
Camera image -> yolov5 -> if squirrel -> fire mechanism and send photo, else do nothing.

"""

import datetime
import subprocess
import time
import argparse
import sys

from tenacity import retry, wait_fixed, retry_if_exception_type

import telegram_bot
import strike


# Running on a pi - can't import torch.
#     Split off the motion detector into another file?
# todo - 1. pi-running motion detector to gather images of squirrels.
#  telegram to send photos to chat on request.
#  auto get sunrise and sunset.
#  test speed of new weights.
#  record the frame rate while running inference on it.


@retry(wait=wait_fixed(60), retry=retry_if_exception_type(AssertionError))
def main(source='',
         weights='yolov5n6_best.pt',
         imgsz=(1280, 1280),
         telegram_bot_mode=True,
         surveillance_mode=False,  # Don't run the strike functions.
         motion_detection=True,
         inference=True,
         pi_mode=False
         ):
    try:
        if pi_mode:
            import pi_motion_detection
            inference = False
            subprocess.Popen(
                ["python3", "~/ironacer/stream.py"],
                stdin=None, stdout=None, stderr=None, close_fds=True)
            # If running on the pi, never has inference on as it can't install torch.
            d = pi_motion_detection.PiMotion(source=source)

        else:
            import find
            subprocess.Popen(["ssh", "pi@ironacer.local", "python3 ~/ironacer/stream.py"],
                             stdin=None, stdout=None, stderr=None, close_fds=True)
            time.sleep(5)

            d = find.StreamDetector(source=source, weights=weights, motion_detection_only=motion_detection, imgsz=imgsz)
        # claymore = strike.Claymore()
        bot = telegram_bot.TelegramBot()
        bot.send_message(f'Starting recoding. Pi_mode: {pi_mode}, Inference: {inference}')
        for path, im, im0s, vid_cap, s in d.stream():
            if motion_detection:
                d.motion_detector(im0s[0])  # Saves motion detected images.

            if inference:
                isSquirrel, inference = d.inference(im, im0s)  # Runs yolov5 inference.

                d.save_train_data(im0s[0], isSquirrel, inference)  # Saves training data (clean images and labels)
                vid_path = d.save_labeled(im0s[0], isSquirrel, inference)  # Saves videos of detected squirrels.

                if isSquirrel and not surveillance_mode:  # Squirrel is present
                    # claymore.detonate()  # Currently just does nothing.
                    pass
                if vid_path is not False and telegram_bot_mode is True:
                    bot.send_video(vid_path=vid_path)
            now = datetime.datetime.now()
            sunset = datetime.datetime(year=now.year, month=now.month, day=now.day, hour=16, minute=37)
            if now > sunset:
                return None
    finally:
        pass
        if pi_mode:
            subprocess.Popen(["pkill -f ~/ironacer/stream.py"],
                             stdin=None, stdout=None, stderr=None, close_fds=True)
        else:
            subprocess.Popen(["ssh", "pi@ironacer.local", "pkill -f ~/ironacer/stream.py"],
                             stdin=None, stdout=None, stderr=None, close_fds=True)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='tcp://ironacer.local:5000')
    parser.add_argument('--surveillance_mode', type=bool, default=False, help='True = run pi surveillance to capture data. ')
    parser.add_argument('--motion_detection', type=bool, default=True, help='Run motion detection')
    parser.add_argument('--inference', type=bool, default=True, help='Run yolo inference or not.')
    parser.add_argument('--weights', type=bool, default='yolov5n6_best.pt', help='File path to yolo weights.pt')
    parser.add_argument('--imgsz', type=bool, default=(1280, 1280), help='tuple of inference image size.')
    parser.add_argument('--telegram_bot_mode', type=bool, default=True, help='Run telegram or not.')
    parser.add_argument('--pi_mode', type=bool, default=False, help='Running on pi or not?')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = arg_parse()
    if len(sys.argv) == 1:  # Run this if from pycharm, otherwise it's command line.
        opt.source = 'http://ironacer.local:8000/stream.mjpg'
        opt.surveillance_mode = True
        opt.motion_detection = True
        opt.inference = False
        opt.pi_mode = False
    main(**vars(opt))

