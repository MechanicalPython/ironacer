#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.

Workflow
Camera image -> yolov5 -> if squirrel -> fire mechanism and send photo, else do nothing.

"""

import time
import argparse
import sys
import os
import cv2


import telegram_bot
import strike
from stream import LoadWebcam


# todo
#  telegram to send photos to chat on request.
#  auto get sunrise and sunset.


def save_results(frame, xyxyl, type):
    """Saves a clean image and the label for that image.
    label = x, y, x, y, label.
    """
    parent_folder = os.path.dirname(__file__)
    if parent_folder == '':
        parent_folder = '.'

    t = str(time.time())
    image_path = f'{parent_folder}/detected/image/{type}_result-{t}.jpg'
    cv2.imwrite(image_path, frame)  # Write image
    label_path = f'{parent_folder}/detected/label/{type}_result-{t}.txt'

    label = ''
    for box in xyxyl:
        box = [str(i) for i in box]
        label = f'{label}{" ".join(box)}\n'

    with open(label_path, 'w') as f:
        f.write(label)


def main(source=0,
         weights='yolov5n6_best.pt',
         imgsz=(1280, 1280),
         telegram_bot_mode=True,
         surveillance_mode=False,  # Don't run the strike functions.
         motion_detection=True,
         inference=True,
         on_mac=False
         ):

    if inference:
        from find import Detector
        yolo = Detector(weights, imgsz)

    if motion_detection:
        from motion_detection import MotionDetection
        motion_detector = MotionDetection(detection_region=[0, 250, 500, 1280])

    if not surveillance_mode:
        claymore = strike.Claymore()

    # Set up the stream and the inference or motion detection classes as needed.
    with LoadWebcam(pipe=source, img_size=imgsz, on_mac=on_mac) as stream:
        for frame in stream:

            if inference:
                is_squirrel, inference_result = yolo.inference(frame)
                xyxy, conf, cls = inference_result   # xyxy is list of 4 items.

                if is_squirrel:  # Save image
                    xyxy.append(conf)  # add conf to xyxy to save it.
                    save_results(frame, xyxy, 'Yolo')

            if motion_detection:
                is_motion, motion_detection_result = motion_detector.detect(frame)  # list of [xyxy, amount_of_motion]
                if is_motion:  # Save image
                    save_results(frame, motion_detection_result, 'Motion')

            if not surveillance_mode:
                pass
                # claymore.detonate()
                # One day, strike.javelin(result)

            # if telegram_bot_mode:
            #     if is_squirrel:
            #         # Send video
            #         pass
            #     if is_motion:
            #         # Send video
            #         pass
            #
            #     if telegram.new_message == 'send_photo':
            #         send_photo(frame)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=0)
    parser.add_argument('--weights', type=str, default='yolov5n6_best.pt', help='File path to yolo weights.pt')
    parser.add_argument('--imgsz', type=tuple, default=(1280, 1280), help='tuple of inference image size.')
    parser.add_argument('--telegram_bot_mode', type=boolean_string, default=True, help='Run telegram or not.')
    parser.add_argument('--surveillance_mode', type=boolean_string, default=False, help='True = do strike')
    parser.add_argument('--motion_detection', type=boolean_string, default=True, help='Run motion detection')
    parser.add_argument('--inference', type=boolean_string, default=True, help='Run yolo inference or not.')
    parser.add_argument('--on_mac', type=boolean_string, default=False, help='True if running on mac.')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = arg_parse()
    if len(sys.argv) == 1:  # Run this if from pycharm, otherwise it's command line.
        opt.surveillance_mode = True
        opt.motion_detection = True
        opt.inference = False
        opt.on_mac = True
    main(**vars(opt))

