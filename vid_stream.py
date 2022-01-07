"""
Streams the camera video via cv2 to get higher resolution images.
"""

import cv2
import sys


def video_stream(og_res, crop_xywh):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Could not open video device")
    # Set properties. Each returns === True on success (i.e. correct resolution)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, og_res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, og_res[1])
    x, y, w, h = crop_xywh
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = frame[y:y+h, x:x+w]
            sys.stdout.write(frame.tobytes())
    cap.release()


if __name__ == '__main__':
    video_stream()

