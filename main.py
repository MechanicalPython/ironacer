"""
Notes:
    Not going to try and account for false negatives. It's too complex. If it turns out that the AI is missing squirrels
    then I'll do something about it.


"""
import argparse
import datetime
import os
import threading
import time
import cv2

import suntime

from ironacer import DETECTION_REGION, YOLO_WEIGHTS, IMGSZ, MOTION_THRESH
from ironacer import strike, telegram_bot, stream, find, motion_detection, utils


# todo
#  run telegram, inference, and motion detection on separate threads to speed it up.
#  Benchmark fps while running yolo on the pi.


class IronAcer:
    """
    Motion detection just used to trigger yolo or not.

    Only saves yolo detected images of squirrels plus previous image and the next 5 images.

    Runs forever with a service: https://www.tomshardware.com/how-to/run-long-running-scripts-raspberry-pi

    """

    def __init__(self,
                 surveillance_mode=False,  # Don't run the strike functions.
                 gather_data=True):  # Keep as the systemctl service expects it.
        self.surveillance_mode = surveillance_mode

        self.yolo = find.Detector(weights=YOLO_WEIGHTS, imgsz=640)

        self.motion_detector = motion_detection.MotionDetection(
            detection_region=DETECTION_REGION, motion_thresh=MOTION_THRESH)
        self.claymore = strike.Claymore()

        self.bot = telegram_bot.TelegramBot()
        self.bot.claymore = self.claymore

        self.sun = suntime.Sun(51.5, -0.1)  # London lat long.

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
                if temp > 85:
                    self.bot.send_message(f'Warning: CPU temperature is {temp}')
            time.sleep(60*60)

    def main(self):
        """
        Main thread for ironacer

        Note: AI is not accurate enough to reliably fire water yet. Keep training it.

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

                for frame in frames:
                    self.bot.latest_frame = frame

                    is_motion, motion_detection_result = self.motion_detector.detect(frame)
                    if is_motion:
                        is_squirrel, inference_result = self.yolo.inference(frame)
                        if is_squirrel:
                            self.claymore.start()
                            vid_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (IMGSZ, IMGSZ))
                            for i in range(0, 10):
                                frame = frames.__next__()
                                vid_writer.write(utils.add_label_to_frame(frame, inference_result, 'Yolo'))
                            vid_writer.release()
                            self.claymore.stop()
                            with open('temp.mp4') as f:
                                self.bot.send_video(f)
                            os.remove('temp.mp4')

                    if not self.is_daytime():
                        self.bot.send_message(self.bot.detected_info())
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
