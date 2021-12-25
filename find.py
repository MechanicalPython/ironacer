#! /usr/local/bin/python3.7

"""
FIND.py - find the exact location of the squirrel in the image in 3d space.
"""

import os
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


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
        """Return path to the latest saved image in runs/detect_from_image/exp*/.jpg"""
        root = 'runs/detect_from_image/'
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

    def detect_from_image(self, img):
        """
        Finds the coordinates of the item most likely to be a squirrel (from top left = 0,0)
        Return image of the box and corrdinates of the squirrel and true/false?
        :param: photo (either file path of rb in RAM)
        :return: df of results.
        """
        results = self.model(img)
        print('hello')
        df = results.pandas().xyxy[0]
        if len(df) > 0:
            results.save()
            picture = self.get_latest_saved_image_path()
            return True, self.results_to_centre_coord(df), picture
        return False, 0, 0


@torch.no_grad()
def detect_stream(weights='best.pt',  # model.pt path(s)
                  source='http://ironacer.local:8000/stream.mjpg',
                  imgsz=(640, 640),  # inference size (height, width)
                  conf_thres=0.25,  # confidence threshold
                  iou_thres=0.45,  # NMS IOU threshold
                  max_det=1000,  # maximum detections per image
                  device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                  classes=None,  # filter by class: --class 0, or --class 0 2 3
                  agnostic_nms=False,  # class-agnostic NMS
                  augment=False,  # augmented inference
                  visualize=False,  # visualize features
                  line_thickness=3,  # bounding box thickness (pixels)
                  hide_labels=False,  # hide labels
                  hide_conf=False,  # hide confidences
                  half=False,  # use FP16 half-precision inference
                  dnn=False,  # use OpenCV DNN for ONNX inference
                  ):
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    # view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '

            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):  # If found a squirrel, this is triggered.
                # det = tensor list of xmin, ymin, xmax, ymax, confidence, class number
                # Rescale boxes from img_size to im0 size, basically normalises it.
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                print(det[:, :4])
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add box to image.
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                im0 = annotator.result()
                cv2.imwrite(f'temp.jpg', im0)
                yield True, 'temp.jpg'
            yield False, None, None


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
    for i in detect_stream(source='http://ironacer.local:8000/stream.mjpg'):
        print(i)
