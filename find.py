#! /usr/local/bin/python3.7

"""
FIND.py - find the exact location of the squirrel in the image in 3d space.
"""

import torch
import os


class Detector:
    """
    darknet_detect() box gives left, top, width, height. left and top gives the top left coordinate
    for the box. The width and height then allows you to work out the next 3 corners of the box.
    """

    def __init__(self):
        self.model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
        self.model.cpu()
        self.model.conf = 0.1

    @staticmethod
    def get_latest_saved_image_path():
        """Return path to the latest saved image in runs/detect/exp*/.jpg"""
        root = 'runs/detect/'
        exps = os.listdir(root)
        exps.sort()
        exp = f'{root}{(exps[len(exps)-1])}/'
        images = os.listdir(exp)
        return [f'{exp}{i}' for i in images if i.endswith('.jpg')]

    @staticmethod
    def results_to_centre_coord(results):
        """Convert results of the model to a list of x,y coordinates for each squirrel found"""
        # pandas().xyxy returns xmin, ymin, xmax, ymax, from top left of image.
        results['confidence'].sort_values()
        xyxy = results.to_dict('index')[0]
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
        Return image of the box and corrdinates of the squirrel and true/false?
        :param: photo (either file path of rb in RAM)
        :return: df of results.
        """
        self.model.
        print(self.model(img))
        # for i in self.model(img):
        #     print(i)
        # results = self.model(img)
        # print('hello')
        # df = results.pandas().xyxy[0]
        # if len(df) > 0:
        #     results.save()
        #     picture = self.get_latest_saved_image_path()
        #     return True, self.results_to_centre_coord(df), picture
        # return False, 0, 0


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
    pass
    # Detector().detect('http://ironacer.local:8000/stream.mjpg')

