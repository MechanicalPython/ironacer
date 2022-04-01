
import cv2


class MotionDetection:
    """
    # Motion detection class: to use basic motion detection to gather images of squirrels,
     based off https://www.geeksforgeeks.org/webcam-motion-detector-python/

    # Attributes
    detection_region
    motion_thresh - 500 is very low, over 4000 is roughly that of an average moving squirrel:
    prev_frame

    ## Methods
    - detect(frame, motion_thresh) -> is_motion: bool, bounding_boxes: list of lists
        Main method that inputs a frame and compared it to the previous frame given to detect motion.
        Bounding boxes are x, y, x, y. (top left, bottom right). Origin is top left of image.
    - motion_region
        Filters out boxes of motion that do not fall within a specified region.

    ## Notes
    A moving object will detect motion when the object moves into frame and when it moves out of the frame, leaving a
    motion detection box around the empty space left by an object.
    """
    def __init__(self, detection_region, motion_thresh):
        """
        Parameters
        -----------
        prev_frame - Stores the previous frame (frame A) for comparison to frame B.
        detection_region - The rectangle that allows motion inside it.
            list of the x, y, x, y coords for top left bottom right rectangle.
            Will miss motion that is larger than the detection_region:
            _________________
            |               |
            |      []       |
            |_______________|
            If the small box is the detection_region, the large box will not be included as all corners are outside
            the detection region.
        """
        self.prev_frame = None
        self.detection_region = detection_region
        self.motion_thresh = motion_thresh

    def detect(self, frame):
        """
        If there is motion between frame A and frame B, this saves frame B and the bounding boxes for that motion
        in a labels file.

        :param frame: np.array from cv2 VideoCapture(0) read().
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
            if amount_of_motion < self.motion_thresh:
                continue  # go to next contour.

            # cv2.boundingRect gives x, y, width of box, height of box. This is then converted to the yolo
            # format: x, y, x, y (top left, bottom right).
            (x, y, w, h) = cv2.boundingRect(contour)  # cv2.boundingRect gives
            bounding_boxes.append([x, y, w+x, h+y, int(amount_of_motion)])  # Convert x, y, x, y for yolo.
        is_motion, bounding_boxes = self._motion_region(bounding_boxes)
        self.prev_frame = frame
        return is_motion, bounding_boxes

    def _motion_region(self, bounding_boxes):
        """Determines if a bounding box is inside of self.detection_region and removes boxes that are not valid.
        return:
            bool (true is box is inside of detection_region
            boxes where bool is true.
        """
        positive_boxes = []
        for box in bounding_boxes:
            x, y, a, b, _ = box  # x, y, a, b = top left, top right
            corners = (x, y), (a, y), (x, b), (a, b)
            for corner in corners:
                cx, cy = corner
                if self._coord_in_rect(cx, cy):
                    positive_boxes.append(box)
                    break  # To stop duplicates if 2 corners are in the detection region.
        if len(positive_boxes) == 0:
            return False, [[None, None, None, None, None]]
        else:
            return True, positive_boxes

    def _coord_in_rect(self, x, y):
        """Returns bool for is a given coordinate falls within self.detection_region."""
        if self.detection_region[0] <= x <= (self.detection_region[2]) and \
                self.detection_region[1] <= y <= (self.detection_region[3]):
            return True
        else:
            return False


