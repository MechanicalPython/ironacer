"""
Stream raw cv2 video, as an array, that other aspects of the program can plug into.

"""
import logging
import os
import time

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
    def __init__(self, pipe='0', capture_size=(4056, 3040), output_img_size=(4056, 3040), stride=32):
        self.capture_size = capture_size
        self.output_img_size = output_img_size

        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        # self.reset_freq = 60 * 60  # Frequency to reset the camera (in seconds).
        self.t = time.time()
        self.cap = None

    def set_camera(self):
        # 0.75 is manual control.
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.read()  # Clear buffer
        time.sleep(1)

        width, height = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        x1 = int((width / 2) - (self.output_img_size[0] / 2))
        y1 = int((height / 2) - (self.output_img_size[1] / 2))
        x2 = int((width / 2) + (self.output_img_size[0] / 2))
        y2 = int((height / 2) + (self.output_img_size[1] / 2))
        self.crop_xyxy = [x1, y1, x2, y2]

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
        ret_val, img = self.cap.read()
        if img is None:
            logging.critical(f'Frame is None. {self.get_all_settings()}')
            return None
        # Crop image to correct size.
        # x1, y1, x2, y2 = self.crop_xyxy
        # img = img[y1:y2, x1:x2]  # y1:y2, x1:x2 where x1y1 it top left and x2y2 is bottom right.

        # if time.time() - self.t > self.reset_freq:
        #     image_path = f'{parent_folder}/detected/image/sample_result-{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")}'
        #     cv2.imwrite(image_path, img)
        #
        #     logging.debug(f'fps: {self.frames_produced / self.reset_freq}')
        #     self.reset_camera()
        #     self.t = time.time()
        return img

    # def reset_camera(self):
    #     self.cap.read()
    #     self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #     logging.debug(f'Exposure: {self.cap.get(cv2.CAP_PROP_EXPOSURE)}')

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
    with LoadWebcam() as stream:
        for img in stream:
            cv2.imwrite('test.jpg', img)
            # show_frame(img)
            break