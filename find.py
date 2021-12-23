#! /usr/local/bin/python3.7

"""
FIND.py - find the exact location of the squirrel in the image in 3d space.
"""

import torch


class Detector:
    """
    darknet_detect() box gives left, top, width, height. left and top gives the top left coordinate
    for the box. The width and height then allows you to work out the next 3 corners of the box.
    """

    def __init__(self):
        self.model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
        self.model.cpu()
        self.model.conf = 0.5

    @staticmethod
    def results_to_centre_coord(results):
        """Convert results of the model to a list of x,y coordinates for each squirrel found"""
        # pandas().xyxy returns xmin, ymin, xmax, ymax, from top left of image.
        results_df = results.pandas().xyxy[0]
        results_df['confidence'].sort_values()
        xyxy = results_df.to_dict('index')[0]
        xmin = xyxy['xmin']
        ymin = xyxy['ymin']
        xmax = xyxy['xmax']
        ymax = xyxy['ymax']
        x = int(xmax) - int(xmin)
        y = int(ymax) - int(ymin)
        return x, y

    def detect(self, img):
        """
        Finds the coordinates of the item most likely to be a squirrel (from top left = 0,0)

        :param: photo (either file path of rb in RAM)
        :return: df of results.
        """
        results = self.model(img)

        return results


def angle_from_center(fov, total_width, object_loc):
    """
    Takes fov and image data to work out on what angle the obejct is from the
    center of the camera.
    :param fov: in total degrees of vision
    :param total_width: in pixels, probably 1080.
    :param object_loc: in pixels
    :return: angle relative from the center of the camera.
    """
    rel_loc = (object_loc - (total_width / 2)) / (total_width / 2)
    angle = rel_loc * fov / 2
    return angle


if __name__ == '__main__':
    d = Detector()
    print(d.detect('test.jpg'))
    print(d.detect('test2.jpg'))
    print(d.detect('test3.jpg'))
    print(d.detect('test4.jpg'))

