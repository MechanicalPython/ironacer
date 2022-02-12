"""
FIND.py - find the exact location of the squirrel in the image in 3d space.
"""

import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import sys
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = Path(os.path.abspath(FILE.parents[0]))  # Absolute path

sys.path.insert(0, f'{ROOT}/yolov5/')  # To allow importing from submodule yolov5.
from yolov5.utils.augmentations import letterbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (non_max_suppression, scale_coords)
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device


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


class Detector:
    """Class to detect and read a stream from a pi camera to then run yolo inference on each frame."""
    def __init__(self, weights='best.pt', imgsz=(1280, 1280), conf_thres=0.25):
        self.weights = weights
        self.imgsz = imgsz  # inference size (height, width)

        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.nosave = False  # do not save images/videos
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.line_thickness = 3  # bounding box thickness (pixels)
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
        self.model.warmup(imgsz=(1, 3, *imgsz), half=self.half)  # warmup
        self.number_of_frames_without_squirrel = 0  # How many frames in a row can be false before resetting the vid
        self.vid_writer = None

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
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
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
        else:
            results = None, None, None
        return isSquirrel, results

    def save_labeled_video(self, frame, isSquirrel, inference):  # Save the video.
        """Should just need a frame and det.
        return None when the video is not ready. Return video path when ready to send out.

        Inputs: isSquirrel, xyxy, confidence, cls,
        """
        save_dir = 'results/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        annotator = Annotator(frame, line_width=self.line_thickness, example=str(self.names))
        if isSquirrel:
            self.number_of_frames_without_squirrel = 10
            for xyxy, conf, cls in inference:
                # Add box to image.
                c = int(cls)  # integer class
                label = (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
        else:
            if self.number_of_frames_without_squirrel > 0:
                self.number_of_frames_without_squirrel -= 1

        vid_done = False
        if not self.nosave:
            im0 = annotator.result()
            if isSquirrel or self.number_of_frames_without_squirrel > 0:  # record video
                if isinstance(self.vid_writer, cv2.VideoWriter):  # Vid_writer has already been created.
                    self.vid_writer.write(im0)
                else:  # Create a new vid_writer and write frame to it.
                    vid_num = len([i for i in os.listdir(save_dir) if i.endswith(".mp4")]) + 1
                    self.current_vid_path = str(f'{save_dir}result-{vid_num}.mp4')
                    fps, w, h = 6, im0.shape[1], im0.shape[0]
                    self.vid_writer = cv2.VideoWriter(self.current_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    self.vid_writer.write(im0)
            else:  # Done recording the video
                if isinstance(self.vid_writer, cv2.VideoWriter):  # If a video has been recorded
                    self.vid_writer.release()  # release previous video writer
                    self.vid_writer = None
                    prev_vid_num = len([i for i in os.listdir(save_dir) if i.endswith(".mp4")])
                    vid_done = str(f'{save_dir}result-{prev_vid_num}.mp4')
        return vid_done


def convert_xyxy_to_yolo_label(frame, xyxy):
    """
    class (0 for squirrel, x_center y_center width height from top right of image and normalised to be 0-1.
    return:
    """

    xmin, ymin, xmax, ymax = xyxy[0]
    im_width, im_height = frame.shape[1], frame.shape[0]
    x_center = (ymin + ((ymax - ymin) / 2)) / im_width
    y_center = (xmin + ((xmax - xmin) / 2)) / im_height
    width = (xmax - xmin) / im_width
    height = (ymax - ymin) / im_height
    return [0, x_center, y_center, width, height]


if __name__ == '__main__':
    from stream import LoadWebcam

    detect = Detector()

    with LoadWebcam() as stream:
        for frame in stream:
            inf = detect.inference(frame)
            print(inf)

