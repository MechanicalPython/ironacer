"""
Wholesale ripped all from yolov5 datasets and supporting docs to just make this work.

"""

import argparse
import os
import time

import cv2


class PiMotion:
    def __init__(self, width, height, imsiz):
        self.prev_frame = None
        self.width = width
        self.height = height
        self.imsiz = imsiz

    def stream(self):
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        x = (self.width - self.imsiz) / 2
        y = (self.height - self.imsiz) / 2
        crop_xywh = (int(x), int(y), self.imsiz, self.imsiz)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = self.crop_frame(frame, crop_xywh)
                yield frame
        yield None

    def motion_detector(self, frame, motion_thresh=500):
        """
        If there is motion between frame A and frame B, this saves frame B and the bounding boxes for that motion
        in a labels file.

        Based on Webcam Motion Detector from https://www.geeksforgeeks.org/webcam-motion-detector-python/

        :param motion_thresh: 500 is low to capture everything, but gets a lot of leaf movement.
        :param frame:
        :return:
        """
        image_path = None
        if frame is None:
            return image_path

        motion = 0  # 0 = no motion, 1 = yes motion.

        og_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting color image to gray_scale image
        frame = cv2.GaussianBlur(frame, (21, 21), 0)

        if self.prev_frame is None:  # Init first frame to gray background.
            self.prev_frame = frame
            return image_path

        # Difference between previous frame and current frame(which is GaussianBlur)
        diff_frame = cv2.absdiff(self.prev_frame, frame)

        # If change in between static background and current frame is greater than 30 it will show white color(255)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # Finding contour of moving object
        cnts, _ = cv2.findContours(thresh_frame.copy(),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in cnts:
            amount_of_motion = cv2.contourArea(contour)
            if amount_of_motion < motion_thresh:  # this is the threshold for motion.
                continue  # go to next contour.

            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            bounding_boxes.append(f'{x} {y} {w} {h} {amount_of_motion}')

        if motion == 1:  # Save the image.
            t = str(time.time())
            image_path = f'{os.path.dirname(__file__)}/motion_detected/image/result-{t}.jpg'
            print(image_path)
            cv2.imwrite(image_path, og_frame)  # Write image
            label_path = f'{os.path.dirname(__file__)}/motion_detected/label/result-{t}.txt'
            with open(label_path, 'w') as f:
                f.write('\n'.join(bounding_boxes))

        self.prev_frame = frame
        return image_path

    @staticmethod
    def crop_frame(frame, crop_xywh):
        x, y, w, h = crop_xywh
        frame = frame[y:y + h, x:x + w]
        _, JPEG = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return JPEG


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=2592)
    parser.add_argument('--height', type=int, default=1944)
    parser.add_argument('--imsiz', type=int, default=1280)
    return parser.parse_args()


if __name__ == '__main__':
    opt = arg_parse()
    d = PiMotion(opt.width, opt.height, opt.imsiz)
    for frame in d.stream():
        d.motion_detector(frame)
