
import argparse
import datetime
import os
import time
import zipfile
import threading

import cv2
import numpy as np
import suntime

from ironacer import strike, telegram_bot, stream, find, motion_detection, utils
from ironacer import ROOT, DETECTION_REGION, YOLO_WEIGHTS, IMGSZ, MOTION_THRESH


# todo
#  run telegram, inference, and motion detection on separate threads to speed it up.
#  Too many photos.
#  Sending the zip file in chunks.


class IronAcer:
    """
    Controls all the sub methods and classes that do the heavy lifting.

    Workflow
    Camera image -> yolov5 -> if squirrel -> fire mechanism and send photo, else do nothing.

    ## Data gathering

    Runs forever with a service: https://www.tomshardware.com/how-to/run-long-running-scripts-raspberry-pi

    """

    def __init__(self,
                 surveillance_mode=False,  # Don't run the strike functions.
                 gather_data=True):
        self.surveillance_mode = surveillance_mode
        self.gather_data = gather_data

        self.yolo = find.Detector(YOLO_WEIGHTS, (IMGSZ, IMGSZ))

        self.motion_detector = motion_detection.MotionDetection(
            detection_region=DETECTION_REGION, motion_thresh=MOTION_THRESH)

        self.bot = telegram_bot.TelegramBot()

        self.sun = suntime.Sun(51.5, -0.1)  # London lat long.

    def start_up(self, frame):
        # Send initial image only at start of the day.
        frame = utils.add_label_to_frame(frame, [DETECTION_REGION])
        self.bot.send_photo(frame)

    def close_down(self):
        self.bot.send_message(f"{len(os.listdir(f'{ROOT}/detected/image/'))} images currently saved")

        # Make zip file and send it.
        zip_file = f"{ROOT}/detected{datetime.datetime.now().strftime('%Y-%m-%d')}.zip"
        zf = zipfile.ZipFile(zip_file, "w")
        for dirname, subdirs, files in os.walk(f'{ROOT}/detected/'):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
                os.remove(os.path.join(dirname, filename))
        zf.close()
        # Stop sending out the zip files, they're too big and causes a memory error crash.
        # self.bot.send_doc(zip_file)
        # os.remove(zip_file)

    def is_daytime(self):
        sunrise = self.sun.get_sunrise_time().replace(tzinfo=None)
        sunset = self.sun.get_local_sunset_time().replace(tzinfo=None)
        return sunrise < datetime.datetime.now() < sunset

    def cpu_temp(self):
        """
        Continuous thread for checking the cpu temp and messaging telegram.
        """
        while True:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                temp = int(f.read().strip()) / 1000
                if temp > 80:
                    self.bot.send_message(f'Warning: CPU temperature is {temp}')
            time.sleep(5)

    def gather_data_motion_detection(self, frame):
        """
        Runs the data gathering with motion detection.
        """
        is_motion, motion_detection_result = self.motion_detector.detect(frame)

        if is_motion:  # There is enough motion, so save the result.
            utils.save_frame(frame, motion_detection_result, 'Motion')

    def find_squirrels(self, frame):
        """Runs the inference for finding squirrels.
        If there is motion:
          If yolo finds a squirrel:
              Run anti-squirrel measures in a thread."""
        is_motion, motion_detection_result = self.motion_detector.detect(frame)
        if is_motion:
            # todo - save results to try and capture data when running inference?
            is_squirrel, inference_result = self.yolo.inference(frame)
            if is_squirrel:
                utils.save_frame(frame, inference_result, 'Yolo')
                return True
        return False

    def main(self):
        """
        Runs yolo inference on frames with enough motion detection, to save power and reduce constant load on the
        pi.
        """
        temp_thread = threading.Thread(target=self.cpu_temp, daemon=True)
        temp_thread.start()
        telegram_thread = threading.Thread(target=self.bot.main, daemon=True)
        telegram_thread.start()
        with stream.LoadCamera(resolution=(IMGSZ, IMGSZ)) as frames:
            while True:
                # If it is nighttime, just go to sleep like you should.
                if not self.is_daytime():
                    time.sleep(60)
                    continue

                # These two lines clear ths buffer (buffer is set to 1) and send the morning message to telegram.
                frames.__next__()  # Clear buffer twice to fix the black image at start up problem.
                frames.__next__()
                self.start_up(frames.__next__())
                for frame in frames:
                    self.bot.latest_frame = frame
                    if self.gather_data:
                        self.gather_data_motion_detection(frame)
                    else:
                        if self.find_squirrels(frame) is True:
                            if self.surveillance_mode is False:
                                strike.threaded_strike()

                    if not self.is_daytime():
                        self.close_down()
                        break


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surveillance_mode', action='store_true', help='Flag to not fire water.')
    parser.add_argument('--gather_data', action='store_true', help='Only gather data with motion detection')
    return parser.parse_args()


if __name__ == '__main__':
    opt = arg_parse()
    IA = IronAcer(**vars(opt))
    # IA.bot.chat_id = 1706759043  # Change it to private chat for testing.
    IA.main()
