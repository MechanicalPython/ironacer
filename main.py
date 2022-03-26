"""
Main.py controls all the sub methods and classes that do the heavy lifting.

Workflow
Camera image -> yolov5 -> if squirrel -> fire mechanism and send photo, else do nothing.

## Data gathering



Runs forever with a service: https://www.tomshardware.com/how-to/run-long-running-scripts-raspberry-pi

"""

import argparse
import datetime
import os
import time
import zipfile
import threading

import cv2
import suntime

import strike
import telegram_bot
from stream import LoadWebcam
from find import Detector
from motion_detection import MotionDetection, add_label_to_frame

from pathlib import Path

# todo
#  run telegram, inference, and motion detection on separate threads to speed it up.


FILE = Path(__file__).resolve()
ROOT = Path(os.path.abspath(FILE.parents[0]))  # Absolute path


class IronAcer:
    def __init__(self,
                 source="0",
                 weights='yolov5n6_best.pt',
                 imgsz=1280,  # Only every going to be square as yolo needs square inputs.
                 detection_region='0,300,1280,800',
                 surveillance_mode=False,  # Don't run the strike functions.
                 gather_data=True):
        self.detection_region = [int(i) for i in detection_region.split(',')]
        self.source = source
        self.weights = weights
        self.imgsz = imgsz
        self.surveillance_mode = surveillance_mode
        self.gather_data = gather_data

        if not gather_data:  # Only load in yolo if needed.
            self.yolo = Detector(weights, (imgsz, imgsz))

        self.motion_detector = MotionDetection(detection_region=self.detection_region)
        self.claymore = strike.Claymore()

        self.bot = telegram_bot.TelegramBot()

        self.sun = suntime.Sun(51.5, -0.1)  # London lat long.
        self.sunrise = self.sun.get_sunrise_time().replace(tzinfo=None)
        self.sunset = self.sun.get_local_sunset_time().replace(tzinfo=None)
        self.now = datetime.datetime.now()

    def start_up(self, frame):
        # Send initial image only at start of the day.
        frame = add_label_to_frame(frame, [self.detection_region])
        self.bot.send_photo(cv2.imencode('.jpg', frame)[1].tobytes())

    def end_of_day_msg(self):
        motion = [i for i in os.listdir(f'{ROOT}/detected/image/') if 'Motion' in i]
        msg = f"{len(motion)} motion detected photos currently saved"
        self.bot.send_message(msg)

        # Make zip file and send it.
        zip_file = f"{ROOT}/detected{self.now.strftime('%Y-%m-%d')}.zip"
        zf = zipfile.ZipFile(zip_file, "w")
        for dirname, subdirs, files in os.walk(f'{ROOT}/detected/'):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
                os.remove(os.path.join(dirname, filename))
        zf.close()
        self.bot.send_doc(zip_file)
        os.remove(zip_file)

    def is_daytime(self):
        self.now = datetime.datetime.now()
        self.sunrise = self.sun.get_sunrise_time().replace(tzinfo=None)
        self.sunset = self.sun.get_local_sunset_time().replace(tzinfo=None)
        return self.sunrise < self.now < self.sunset

    def save_results(self, frame, xyxyl, type):
        """Saves a clean image and the label for that image.
        label = x, y, x, y, label.
        xyxyl = [[x, y, x, y, l], ..]

        Can convert the yolo [[xyxy, confidence, cls], ..] if type is yolo.
        """
        if type == 'Yolo':
            labels = []  # Convert yolo results into cv2 labels.
            for result in xyxyl:
                xyxy, conf, cls = result  # xyxy is list of 4 items.
                xyxy.append(conf)  # add conf to xyxy to save it.
                labels.append(xyxy)
            xyxyl = labels

        t = str(self.now.strftime('%Y-%m-%d %H-%M-%S-%f'))
        image_path = f'{ROOT}/detected/image/{type}_result-{t}.jpg'
        cv2.imwrite(image_path, frame)  # Write image
        label_path = f'{ROOT}/detected/label/{type}_result-{t}.txt'

        label = ''
        for box in xyxyl:
            box = [str(i) for i in box]
            label = f'{label}{" ".join(box)}\n'

        with open(label_path, 'w') as f:
            f.write(label)

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
        # motion_detection_result = [[xyxy, amount_of_motion], ..]

        if is_motion:  # There is enough motion, so save the result.
            self.save_results(frame, motion_detection_result, 'Motion')

    def find_squirrels(self, frame):
        """Runs the inference for finding squirrels.
        If there is motion:
          If yolo finds a squirrel:
              Run anti-squirrel measures in a thread."""
        is_motion, motion_detection_result = self.motion_detector.detect(frame)
        if is_motion:
            # todo - save results to try and capture data when running inference?
            self.save_results(frame, motion_detection_result, 'Motion')
            is_squirrel, inference_result = self.yolo.inference(frame)
            if is_squirrel:
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
        with LoadWebcam(pipe=self.source, output_img_size=(self.imgsz, self.imgsz)) as stream:
            while True:
                # If it is nighttime, just go to sleep like you should.
                if not self.is_daytime():
                    time.sleep(60)
                    continue

                # These two lines clear ths buffer (buffer is set to 1) and send the morning message to telegram.
                stream.__next__()  # Clear buffer.
                self.start_up(stream.__next__())
                for frame in stream:
                    self.bot.latest_frame = cv2.imencode('.jpg', frame)[1].tobytes()
                    if self.gather_data:
                        self.gather_data_motion_detection(frame)
                    else:
                        if self.find_squirrels(frame) is True:
                            if self.surveillance_mode is False:
                                threading.Thread(target=self.claymore.detonate).start()

                    if not self.is_daytime():
                        print('End of day')
                        self.end_of_day_msg()
                        break


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="0")
    parser.add_argument('--weights', type=str, default='yolov5n6_best.pt', help='File path to yolo weights.pt')
    parser.add_argument('--imgsz', type=int, default=1280, help='Square image size.')
    parser.add_argument('--detection_region', type=str, default='0,300,1280,800', help='Set detection region:x,y,x,y')
    parser.add_argument('--surveillance_mode', type=boolean_string, default=False, help='True = do strike')
    parser.add_argument('--gather_data', action='store_true', help='Only gather data with motion detection')
    return parser.parse_args()


if __name__ == '__main__':
    opt = arg_parse()
    IA = IronAcer(**vars(opt))
    # IA.bot.chat_id = 1706759043  # Change it to private chat for testing.
    IA.main()
