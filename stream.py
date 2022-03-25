"""
Stream raw cv2 video, as an array, that other aspects of the program can plug into.

"""
import logging
import os
import time
import argparse

import cv2
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = Path(os.path.abspath(FILE.parents[0]))  # Absolute path

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=f'{ROOT}/detected/stream.log')


def show_frame(frame, rects=None):
    """

    :param frame:
    :param rects: list of [x, y, w, h, label] to put up labels.
    :return:
    """
    if rects is not None:
        for rect in rects:
            x, y, w, h, label = rect
            x, y, w, h, label = int(x), int(y), int(w), int(h), str(label)
            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Motion Box", frame)
    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        return False


class LoadWebcam:
    """
    Taken and modified from yolov5/utils LoadWebcam.
    Returns just the image, the augmentation needed for inference is done by find.py.
    hq camera - 4056 x 3040 pixels max resolution.
    """
    def __init__(self, pipe='0', capture_size=(1280, 1280), output_img_size=(1280, 1280), stride=32):
        self.capture_size = capture_size
        self.check_resolution()
        self.output_img_size = output_img_size

        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        # self.reset_freq = 60 * 60  # Frequency to reset the camera (in seconds).
        self.t = time.time()
        self.cap = None

    def set_camera(self):
        # 0.75 is manual control.
        self.cap = cv2.VideoCapture(self.pipe)   # For pi0 - VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 is automatic exposure. 0.75 is manual control.
        self.cap.read()  # Clear buffer
        time.sleep(1)

    def __enter__(self):
        print('start camera')
        self.set_camera()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        # Read frame
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.75)  # Hopefully running this will adjust the exposure for each frame.
        ret_val, img = self.cap.read()
        if img is None:
            logging.critical(f'Frame is None. {self.get_all_settings()}')
            return None
        return img

    @staticmethod
    def digital_crop(frame, x1, y1, x2, y2):
        """Digital crop. x1 and y1 is the top left of the image. x2 and y2 is the bottom right."""
        frame = frame[y1:y2, x1:x2]  # y1:y2, x1:x2 where x1y1 it top left and x2y2 is bottom right.
        return frame

    def check_resolution(self):
        """The camera's block size is 32x16 so any image data provided to a renderer must have a width which is a
        multiple of 32, and a height which is a multiple of 16."""
        if self.capture_size[0] % 32 != 0:
            raise Exception('Width must be multiple of 32')
        if self.capture_size[1] % 16 != 0:
            raise Exception('Height must be multiple of 16')

    def get_all_settings(self):
        return f"""
        CAP_PROP_MODE:  {str(self.cap.get(cv2.CAP_PROP_MODE))}
        CAP_PROP_FPS:  {str(self.cap.get(cv2.CAP_PROP_FPS))}
        CAP_PROP_CONTRAST:  {str(self.cap.get(cv2.CAP_PROP_CONTRAST))}
        CAP_PROP_GAIN:  {str(self.cap.get(cv2.CAP_PROP_GAIN))}
        CAP_PROP_FRAME_WIDTH:  {str(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}
        CAP_PROP_FRAME_HEIGHT:  {str(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))}
        CAP_PROP_POS_FRAMES:  {str(self.cap.get(cv2.CAP_PROP_POS_FRAMES))}
        CAP_PROP_EXPOSURE:  {str(self.cap.get(cv2.CAP_PROP_EXPOSURE))}"""


# max - 3280 × 2464 pixels
# 1-15 fps - 2592 x 1944

if __name__ == '__main__':
    with LoadWebcam(pipe='0', output_img_size=(1280, 1280)) as stream:
        for frame in stream:
            print(type(frame))

