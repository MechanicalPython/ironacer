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
import os
import cv2


from tenacity import retry, wait_fixed, retry_if_exception_type

import telegram_bot
import strike
import stream
import find
import pi_motion_detection

# todo
#  telegram to send photos to chat on request.
#  auto get sunrise and sunset.


def save_results(frame, xyxy, label, type):
    """Saves a clean image and the label for that image"""
    parent_folder = __file__
    if parent_folder == '':
        parent_folder = '.'

    t = str(time.time())
    image_path = f'{parent_folder}/detected/image/{type}_result-{t}.jpg'
    cv2.imwrite(image_path, frame)  # Write image
    label_path = f'{parent_folder}/detected/label/{type}_result-{t}.txt'

    with open(label_path, 'w') as f:
        f.write(f'{" ".join(xyxy)} {label}')


def main(source='',
         weights='yolov5n6_best.pt',
         imgsz=(1280, 1280),
         telegram_bot_mode=True,
         surveillance_mode=False,  # Don't run the strike functions.
         motion_detection=True,
         inference=True,
         ):

    # Set up the stream and the inference or motion detection classes as needed.
    streamer = stream.Streamer(width=2592, height=1944, imsiz=1280)
    if inference:
        yolo = find.Detector(weights, imgsz)

    if motion_detection:
        motion_detector = pi_motion_detection.MotionDetection(
            width=2592, height=1944, imsiz=1280, detection_region=[0, 250, 500, 1280])

    while True:
        is_squirrel = False
        is_motion = False

        frame = streamer.get_frame()
        if inference:
            is_squirrel, inference_result = yolo.inference(frame)
            xyxy, conf, cls = inference_result   # xyxy is list of 4 items.

            if is_squirrel:# Save image
                save_results(frame, xyxy, f'{conf}%_Squirrel', 'Yolo')

        if motion_detection:
            is_motion, motion_detection_result = motion_detector.detect(frame)  # list of [xyxy, amount_of_motion]
            xyxy, amount_of_motion = motion_detection_result
            if is_motion:# Save image
                save_results(frame, xyxy, amount_of_motion, 'Motion')


        if not surveillance_mode:
            strike.claymore()
            # One day, strike.javelin(result)

        if telegram_bot_mode:
            if is_squirrel:
                # Send video
                pass
            if is_motion:
                # Send video
                pass

            if telegram.new_message == 'send_photo':
                send_photo(frame)




def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='http://ironacer.local:8000/stream.mjpg')
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

