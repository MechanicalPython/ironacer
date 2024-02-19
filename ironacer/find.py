"""
Notes
running at 640x640 detect.py analyses an image around 0.2-0.3 seconds on my mac mini.
At 1280x1280 that's at 0.7-1.2 seconds. There was a drop in accuracy at 640 but mostly in including too many things
as squirrels. The pi may have to run at 640 detection in order to maintain performance. See runs/detect/exp3 and 4
for 1280 and 640 respectively.


"""

import torch
import torch.backends.cudnn as cudnn
import sys
import numpy as np

from ironacer import ROOT

sys.path.insert(0, f'{ROOT}/yolov5/')  # To allow importing from submodule yolov5.
from yolov5.utils.augmentations import letterbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (non_max_suppression, scale_coords)
from yolov5.utils.torch_utils import select_device


class SpatialDetector:
    pass


class Detector:
    """Class to detect and read a stream from a pi camera to then run yolo inference on each frame."""

    def __init__(self, weights='best.pt', imgsz=1280, conf_thres=0.25):
        self.weights = weights
        self.imgsz = (imgsz, imgsz)  # inference size (height, width)

        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.device = select_device(self.device)
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn)
        self.stride, self.names, self.pt, jit, self.onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.half &= (self.pt or jit or engine) and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA

        if self.pt or jit:
            self.model.model.half() if self.half else self.model.model.float()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

    @torch.no_grad()
    def inference(self, img0):
        """
        Pulled from yolov5/detect.py

        xyxy = top left and bottom right of the bounding box.

        :param img0: the result of cv2.imread()
        :return: isSquirrel: bool, results: [[xyxy, confidence, cls], ..] for each object found.
        """
        # Taken from yolov5/utils/datasets.py LoadImages class.
        im = letterbox(img0, self.imgsz[0], stride=self.stride)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        # Process predictions
        pred = pred[0]

        isSquirrel = False
        results = []

        if len(pred):  # If found a squirrel, this is triggered.
            isSquirrel = True
            # det = tensor list of xmin, ymin, xmax, ymax, confidence, class number

            # Rescale boxes from img_size to im0 size, basically normalises it.
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(pred):  # the *xyxy is to take the first 4 items as the coords.
                confidence = conf.item()
                xyxy = [i.item() for i in xyxy]  # Convert from [tensor(x), ..] to [x, ..]
                results.append([xyxy, confidence, cls])
        return isSquirrel, results


