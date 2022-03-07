"""
Script that can be run on the pi to detect motion for capturing data for yolo training.

"""
import os
import cv2


class MotionDetection:
    def __init__(self):
        """
        :rtype: object
        """
        self.parent_folder = os.path.dirname(__file__)
        if self.parent_folder == '':
            self.parent_folder = '.'

        self.prev_frame = None

    def detect(self, frame, motion_thresh=500):
        """
        If there is motion between frame A and frame B, this saves frame B and the bounding boxes for that motion
        in a labels file.

        Based on Webcam Motion Detector from https://www.geeksforgeeks.org/webcam-motion-detector-python/

        Bounding boxes are x, y, width, height. Origin is top left of the image.
        :param motion_thresh: 500 is low to capture everything, but gets a lot of leaf movement.
        :param frame:
        :return: is_motion, list of [x, y, x, y, amount of motion]
        """
        if frame is None:
            return False, [[None, None, None, None, None]]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting color image to gray_scale image
        frame = cv2.GaussianBlur(frame, (21, 21), 0)

        if self.prev_frame is None:  # Init first frame to gray background.
            self.prev_frame = frame
            return None, [[None, None, None, None, None]]

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
            (x, y, w, h) = cv2.boundingRect(contour)
            bounding_boxes.append([x, y, x+w, y+h, amount_of_motion])  # Convert to top left and top right coords for compatibility with yolo convention.
        self.prev_frame = frame
        return is_motion, bounding_boxes


if __name__ == '__main__':
    from stream import LoadWebcam
    motion_detector = MotionDetection()

    with LoadWebcam() as stream:
        for frame in stream:
            is_motion, results = motion_detector.detect(frame)  # results = [[[x, y, x, y], motion],.. ]
            rectangles = [[0, 250, 500, 1280, 'DETECT']]
            if results is None:
                continue
            for xyxy, motion in results:
                xyxy.append(motion)
                rectangles.append(xyxy)

