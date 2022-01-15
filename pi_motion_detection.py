"""
Script that can be run on the pi to detect motion for capturing data for yolo training.

"""

import argparse
import os
import time

import cv2

# todo - capture motion only in certain areas of the image.


class PiMotion:
    def __init__(self, width, height, imsiz, on_mac=False, save_images=True):
        self.prev_frame = None
        self.reset_freq = 5*60  # Frequency to reset the camera (in seconds).
        self.width = width
        self.height = height
        self.imsiz = imsiz
        self.on_mac = on_mac

        self.set_video()  # All parameters for the video are to go in this.

        x = (self.width - self.imsiz) / 2
        y = (self.height - self.imsiz) / 2
        self.crop_xywh = (int(x), int(y), self.imsiz, self.imsiz)
        self.save_images = save_images

    def set_video(self):
        if self.on_mac:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def stream(self):
        t = time.time()
        try:
            while True:
                if time.time() - t > self.reset_freq:
                    self.set_video()
                    t = time.time()
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # frame = self.crop_frame(frame, self.crop_xywh)
                        yield frame
        finally:
            self.cap.release()

    def _show_motion_live(self):
        for frame in d.stream():
            frame, bounding_box = self.motion_detector(frame)
            if frame is None:
                continue
            for label in bounding_box:
                x, y, w, h, amount_of_motion = label.split(' ')
                x, y, w, h, amount_of_motion = int(x), int(y), int(w), int(h), str(amount_of_motion)
                # making green rectangle around the moving object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, amount_of_motion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # cv2.imshow("Gray Frame", gray)
            # cv2.imshow("Difference Frame", diff_frame)
            # cv2.imshow("Threshold Frame", thresh_frame)

            cv2.imshow("Motion Box", frame)
            key = cv2.waitKey(1)
            # if q entered whole process will stop
            if key == ord('q'):
                break

    def save_motion_image(self, frame, bounding_boxes):
        """Save the motion detected image. """
        t = str(time.time())
        parent_folder = os.path.dirname(__file__)
        if parent_folder == '':
            parent_folder = '.'
        print(parent_folder)
        image_path = f'{parent_folder}/motion_detected/image/result-{t}.jpg'
        cv2.imwrite(image_path, frame)  # Write image
        label_path = f'{parent_folder}/motion_detected/label/result-{t}.txt'
        with open(label_path, 'w') as f:
            f.write('\n'.join(bounding_boxes))

    def motion_detector(self, frame, motion_thresh=500):
        """
        If there is motion between frame A and frame B, this saves frame B and the bounding boxes for that motion
        in a labels file.

        Based on Webcam Motion Detector from https://www.geeksforgeeks.org/webcam-motion-detector-python/

        :param motion_thresh: 500 is low to capture everything, but gets a lot of leaf movement.
        :param frame:
        :return:
        """
        if frame is None:
            return None, None

        motion = 0  # 0 = no motion, 1 = yes motion.

        og_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting color image to gray_scale image
        frame = cv2.GaussianBlur(frame, (21, 21), 0)

        if self.prev_frame is None:  # Init first frame to gray background.
            self.prev_frame = frame
            return None, None

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

        if motion == 1 and self.save_images:  # Save the image.
            self.save_motion_image(og_frame, bounding_boxes)

        self.prev_frame = frame
        return og_frame, bounding_boxes

    @staticmethod
    def crop_frame(frame, crop_xywh):
        x, y, w, h = crop_xywh
        frame = frame[y:y + h, x:x + w]
        # _, JPEG = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return frame


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=2592)
    parser.add_argument('--height', type=int, default=1944)
    parser.add_argument('--imsiz', type=int, default=1280)
    parser.add_argument('--on_mac', type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    opt = arg_parse()
    opt.on_mac = True
    d = PiMotion(opt.width, opt.height, opt.imsiz, opt.on_mac)
    for frame in d.stream():
        d.motion_detector(frame)
