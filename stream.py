"""
Stream raw cv2 video, as an array, that other aspects of the program can plug into.

"""

import cv2
import argparse
import time


class Streamer:
    def __init__(self, width, height, imsiz, on_mac=False):
        self.reset_freq = 5*60  # Frequency to reset the camera (in seconds).
        self.width = width
        self.height = height
        self.imsiz = imsiz
        self.on_mac = on_mac

        self.set_video()  # All parameters for the video are to go in this.

    def set_video(self):
        if self.on_mac:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        t = time.time()
        try:
            while True:
                if time.time() - t > self.reset_freq:
                    self.set_video()
                    t = time.time()
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.resize(frame, (self.width, self.height))
                        yield frame
        finally:
            self.cap.release()


def show_frame(frame, rects=None):
    """

    :param frame:
    :param rect: list of [x, y, w, h, label] to put up labels.
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


# max - 3280 × 2464 pixels
# 1-15 fps - 2592 x 1944


if __name__ == '__main__':
    stream = Streamer(width=2592, height=1944, imsiz=1280)
    while True:
        frame = stream.stream()
        show_frame(frame)




