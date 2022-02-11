#! /usr/local/bin/python3.7

"""
Main.py controls all the sub methods and classes that do the heavy lifting.

Workflow
Camera image -> yolov5 -> if squirrel -> fire mechanism and send photo, else do nothing.



"""

import argparse
import datetime
import os
import sys
import time
import zipfile

import cv2
import suntime

import strike
import telegram_bot
from stream import LoadWebcam
from find import Detector
from motion_detection import MotionDetection


# todo
#  run telegram, inference, and motion detection on seperate threads to speed it up.

# Set as global variable.
parent_folder = os.path.dirname(__file__)
if parent_folder == '':
    parent_folder = '.'


class IronAcer:
    def __init__(self,
                 source="0",
                 weights='yolov5n6_best.pt',
                 imgsz=1280,  # Only every going to be square as yolo needs square inputs.
                 detection_region='0,350,1280,600',
                 telegram_bot_mode=True,
                 surveillance_mode=False,  # Don't run the strike functions.
                 motion_detection=True,
                 inference=True):
        self.detection_region = [int(i) for i in detection_region.split(',')]
        self.source = source
        self.weights = weights
        self.imgsz = imgsz
        self.telegram_bot_mode = telegram_bot_mode
        self.surveillance_mode = surveillance_mode
        self.motion_detection = motion_detection
        self.inference = inference
        if self.inference:
            self.yolo = Detector(weights, imgsz)

        if self.motion_detection:
            self.motion_detector = MotionDetection(detection_region=self.detection_region)

        if not surveillance_mode:
            self.claymore = strike.Claymore()

        self.bot = telegram_bot.TelegramBot()

        self.sun = suntime.Sun(51.5, -0.1)  # London lat long.
        self.sunrise = self.sun.get_sunrise_time().replace(tzinfo=None)
        self.sunset = self.sun.get_local_sunset_time().replace(tzinfo=None)

    def start_up(self, frame):
        # Send initial image only at start of the day.
        frame = self.add_label_to_frame(frame, [self.detection_region])
        self.bot.send_photo(cv2.imencode('.jpg', frame)[1].tobytes())

    def close_down(self):
        motion = [i for i in os.listdir(f'{parent_folder}/detected/image/') if 'Motion' in i]
        msg = f"{len(motion)} motion detected photos currently saved"
        self.bot.send_message(msg)
        self.send_images()

    def inferencer(self, frame):
        is_squirrel, inference_result = self.yolo.inference(frame)
        xyxy, conf, cls = inference_result  # xyxy is list of 4 items.

        if is_squirrel:  # Save image
            xyxy.append(conf)  # add conf to xyxy to save it.
            self.save_results(frame, xyxy, 'Yolo')
            # self.bot.send_video()  # todo - this.

    def motion_detectoriser(self, frame):
        is_motion, motion_detection_result = self.motion_detector.detect(frame)  # list of [xyxy, amount_of_motion]
        if is_motion:  # Save image
            self.save_results(frame, motion_detection_result, 'Motion')

    def main(self):
        """
        Start via cron job at 2am.
        Sleep till sunrise.
        Take photos
        At sunset close down and turn off.
        """
        with LoadWebcam(pipe=self.source, output_img_size=self.imgsz) as stream:
            if datetime.datetime.now() < self.sunrise:
                time.sleep(self.sunrise - datetime.datetime.now())

            for frame in stream:
                self.start_up(frame)

                now = datetime.datetime.now()

                if now > self.sunset:  # Sunset so close down.
                    self.close_down()
                    break

                if self.inference:
                    self.inferencer(frame)

                if self.motion_detection:
                    self.motion_detectoriser(frame)

                if self.surveillance_mode is False:
                    # claymore.detonate()
                    # One day, strike.javelin(result)
                    pass

    @staticmethod
    def save_results(frame, xyxyl, type):
        """Saves a clean image and the label for that image.
        label = x, y, x, y, label.
        """
        t = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f'))
        image_path = f'{parent_folder}/detected/image/{type}_result-{t}.jpg'
        cv2.imwrite(image_path, frame)  # Write image
        label_path = f'{parent_folder}/detected/label/{type}_result-{t}.txt'

        label = ''
        for box in xyxyl:
            box = [str(i) for i in box]
            label = f'{label}{" ".join(box)}\n'

        with open(label_path, 'w') as f:
            f.write(label)

    @staticmethod
    def add_label_to_frame(frame, xyxyl):
        """
        xyxyl = [[x, y, x, y, label], ] top left, bottom right.
        """
        for label in xyxyl:
            if None in label:
                continue
            if len(label) == 4:
                label.append(' ')
            x, y, x2, y2, amount_of_motion = label
            x, y, x2, y2, amount_of_motion = int(x), int(y), int(x2), int(y2), str(amount_of_motion)
            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, amount_of_motion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return frame

    def send_images(self):
        """Sends zip of the days images at end of the day."""
        # Zip folder
        zip_file = f"{parent_folder}/detected{datetime.datetime.now().strftime('%Y-%m-%d')}.zip"
        zf = zipfile.ZipFile(zip_file, "w")
        for dirname, subdirs, files in os.walk(f'{parent_folder}/detected/'):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
                os.remove(os.path.join(dirname, filename))
        zf.close()
        self.bot.send_doc(zip_file)
        os.remove(zip_file)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="0")
    parser.add_argument('--weights', type=str, default='yolov5n6_best.pt', help='File path to yolo weights.pt')
    parser.add_argument('--imgsz', type=int, default=1280, help='Square image size.')
    parser.add_argument('--detection_region', type=str, default='0,350,1280,600', help='Set detection region:x,y,x,y')
    parser.add_argument('--telegram_bot_mode', type=boolean_string, default=True, help='Run telegram or not.')
    parser.add_argument('--surveillance_mode', type=boolean_string, default=False, help='True = do strike')
    parser.add_argument('--motion_detection', type=boolean_string, default=True, help='Run motion detection')
    parser.add_argument('--inference', type=boolean_string, default=True, help='Run yolo inference or not.')
    return parser.parse_args()


if __name__ == '__main__':
    opt = arg_parse()
    if len(sys.argv) == 1:  # Run this if from pycharm, otherwise it's command line.
        opt.imgsz = 720
        opt.detection_region = '400,400,500,500'
        opt.surveillance_mode = True
        opt.motion_detection = True
        opt.inference = False

    IA = IronAcer(**vars(opt))
    # IA.bot.chat_id = 1706759043  # Change it to private chat for testing.
    IA.main()

