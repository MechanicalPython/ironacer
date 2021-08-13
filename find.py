#! /usr/local/bin/python3.7

"""
FIND.py - find the exact location of the squirrel in the image in 3d space.
"""

import cv2


class Detector:
    """
    darknet_detect() box gives left, top, width, height. left and top gives the top left coordinate
    for the box. The width and height then allows you to work out the next 3 corners of the box.
    """
    def __init__(self):
        self.net = cv2.dnn_DetectionModel('yolo/custom_data/cfg/yolov4-tiny-custom.cfg',
                                          'yolo/custom_data/yolov4-tiny-custom_last.weights')
        self.net.setInputSize(704, 704)
        self.net.setInputScale(1.0/255)
        self.net.setInputSwapRB(True)

        with open('yolo/custom_data/custom.names', 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')

    @staticmethod
    def _box_to_coord(box):
        """Convert left, top, width, height to (top left, top right, bottom right, bottom left)
        Coordinate origin is top left of the picture."""
        left, top, width, height = box
        # left = x axis. top = y axis
        top_left = (left, top)
        top_right = (left+width, top)
        bottom_right = (left + width, top + height)  # addition as origin top left of picture
        bottom_left = (left, top + height)
        return top_left, top_right, bottom_right, bottom_left

    def darknet_detect(self, photo):
        """
        :param: photo (either file path of rb in RAM)
        :return: [class, confidence, boxes] for each found object.
        """
        # photo = cv2.imshow(photo)
        classes, confidence, boxes = self.net.detect(photo)
        if len(classes) == 0:
            return False
        return [[classId, confidence, box] for classId, confidence, box in zip(classes.flatten(), confidence.flatten(), boxes)]

    def save_image(self, objects, photo, save_loc):
        if objects is False:
            photo = cv2.imread(photo)
            cv2.imwrite(save_loc, photo)
        else:
            photo = cv2.imread(photo)
            for classId, confidence, box in objects:
                label = f'{self.names[classId]}: {confidence:.3}'
                text_scale = 3
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, text_scale, 1)
                top_left, top_right, bottom_right, bottom_left = self._box_to_coord(box)
                cv2.rectangle(photo, top_left, bottom_right, color=(0, 255, 0), thickness=5)  # Green box around object
                cv2.rectangle(photo, top_left, (top_left[0] + labelSize[0], top_left[1] + labelSize[1] + baseLine), (255, 255, 255), cv2.FILLED)
                cv2.putText(photo, label, (top_left[0], top_left[1] + labelSize[1] + baseLine), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0), thickness=3)
            cv2.imwrite(save_loc, photo)
        return save_loc


def angle_from_center(fov, total_width, object_loc):
    """
    Takes fov and image data to work out on what angle the obejct is from the
    center of the camera.
    :param fov: in total degrees of vision
    :param total_width: in pixels, probably 1080.
    :param object_loc: in pixels
    :return: angle relative from the center of the camera.
    """
    rel_loc = (object_loc - (total_width/2)) / (total_width/2)
    angle = rel_loc * fov/2
    return angle


if __name__ == '__main__':
    file = 'yolo/custom_data/test.jpg'
    screen = Detector()
    objects = screen.darknet_detect(file)
    screen.save_image(objects, file, 'test.jpg')
