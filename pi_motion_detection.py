"""
Script that can be run on the pi to detect motion for capturing data for yolo training.

"""
import os
import cv2


class MotionDetection:
    def __init__(self, detection_region):
        """

        :rtype: object
        """
        self.parent_folder = os.path.dirname(__file__)
        if self.parent_folder == '':
            self.parent_folder = '.'

        self.prev_frame = None
        self.detection_region = detection_region
        self.reset_freq = 5*60  # Frequency to reset the camera (in seconds).

    def detect(self, frame, motion_thresh=500):
        """
        If there is motion between frame A and frame B, this saves frame B and the bounding boxes for that motion
        in a labels file.

        Based on Webcam Motion Detector from https://www.geeksforgeeks.org/webcam-motion-detector-python/

        Bounding boxes are x, y, width, height. Origin is top left of the image.
        :param motion_thresh: 500 is low to capture everything, but gets a lot of leaf movement.
        :param frame:
        :return: is_motion, list of [xyxy, amount_of_motion].  xyxy is list of 4 items.
        """
        if frame is None:
            return False, None

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
            (x, y, w, h) = cv2.boundingRect(contour)
            xyxy = [x, y, x+w, y+h]  # Convert to top left and top right coords for compatibility with yolo convention.
            bounding_boxes.append([xyxy, amount_of_motion])
        is_motion, bounding_boxes = self.motion_region(bounding_boxes)
        self.prev_frame = frame
        return is_motion, bounding_boxes

    def motion_region(self, bounding_boxes):
        """Set a rectangle where motion can be detected.
        If any part of the motion box is in the detection region it'll be counted. """
        positive_boxes = []
        for box in bounding_boxes:
            xyxy, _ = box
            x, y, a, b = xyxy  # x, y, a, b = top left, top right
            corners = (x, y), (a, y), (x, b), (a, b)
            for corner in corners:
                cx, cy = corner
                if self.coord_in_rect(cx, cy):
                    positive_boxes.append(box)
                    break  # To stop duplicates if 2 corners are in the detection region.
        if len(positive_boxes) == 0:
            return False, positive_boxes
        else:
            return True, positive_boxes

    def coord_in_rect(self, x, y):
        """
        x, y is coordinage from origin of top left of image. Returns bool.
        allowed_rectangle = x, y, w, h
        """
        if self.detection_region[0] <= x <= (self.detection_region[0] + self.detection_region[2]) and \
                self.detection_region[1] <= y <= (self.detection_region[1] + self.detection_region[3]):
            return True
        else:
            return False


if __name__ == '__main__':
    from stream import LoadWebcam, show_frame
    motion_detector = MotionDetection(detection_region=[0, 250, 500, 1280])

    with LoadWebcam() as stream:
        for frame in stream:
            is_motion, results = motion_detector.detect(frame)  # results = [[[x, y, x, y], motion],.. ]
            rectangles = [[0, 250, 500, 1280, 'DETECT']]
            if results is None:
                continue
            for xyxy, motion in results:
                xyxy.append(motion)
                rectangles.append(xyxy)
            show_frame(frame, rects=rectangles)

