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

    def detect(self, frame, motion_thresh=1000):
        """
        If there is motion between frame A and frame B, this saves frame B and the bounding boxes for that motion
        in a labels file.

        Based on Webcam Motion Detector from https://www.geeksforgeeks.org/webcam-motion-detector-python/

        ## Bounding boxes are x, y, x, y. (top left, bottom right). Origin is top left of image.
        :param motion_thresh: 500 is very low, over 4000 is roughly that of an average squirrel.
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

            # cv2.boundingRect gives x, y, width of box, height of box. This is then converted to the yolo
            # format: x, y, x, y (top left, bottom right).
            (x, y, w, h) = cv2.boundingRect(contour)  # cv2.boundingRect gives
            bounding_boxes.append([x, y, w+x, h+y, amount_of_motion])  # Convert x, y, x, y for yolo.
        is_motion, bounding_boxes = self.motion_region(bounding_boxes)
        self.prev_frame = frame
        return is_motion, bounding_boxes

    def motion_region(self, bounding_boxes):
        """Set a rectangle where motion can be detected.
        If any part of the motion box is in the detection region it'll be counted. """
        positive_boxes = []
        for box in bounding_boxes:
            x, y, a, b, _ = box  # x, y, a, b = top left, top right
            corners = (x, y), (a, y), (x, b), (a, b)
            for corner in corners:
                cx, cy = corner
                if self.coord_in_rect(cx, cy):
                    positive_boxes.append(box)
                    break  # To stop duplicates if 2 corners are in the detection region.
        if len(positive_boxes) == 0:
            return False, [[None, None, None, None, None]]
        else:
            return True, positive_boxes

    def coord_in_rect(self, x, y):
        """
        x, y is coordinage from origin of top left of image. Returns bool.
        detection region is in top left bottom right xyxy coords.
        """
        if self.detection_region[0] <= x <= (self.detection_region[2]) and \
                self.detection_region[1] <= y <= (self.detection_region[3]):
            return True
        else:
            return False


def add_label_to_frame(frame, xyxyl):
    """
    xyxyl = [[x, y, x, y, label], ] top left, bottom right.
    """
    for label in xyxyl:
        if None in label:
            continue
        if len(label) == 4:
            label.append(' ')
        x, y, x2, y2, amount_of_motion = label
        x, y, x2, y2, amount_of_motion = int(x), int(y), int(x2), int(y2), str(amount_of_motion)
        # making green rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, amount_of_motion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame


if __name__ == '__main__':
    from stream import LoadWebcam
    motion_detector = MotionDetection(detection_region=[0, 250, 500, 1280])

    with LoadWebcam() as stream:
        for frame in stream:
            is_motion, results = motion_detector.detect(frame)  # results = [[x, y, x, y, motion],.. ]
            rectangles = [[0, 250, 500, 1280, 'DETECT']]
            if results is None:
                continue
            frame = add_label_to_frame(frame, results)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

