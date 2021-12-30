"""
FIND.py - find the exact location of the squirrel in the image in 3d space.
"""

import os
import cv2
import torch
import torch.backends.cudnn as cudnn

import sys
sys.path.insert(0, 'yolov5/')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadStreams
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords)
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


class StreamDetector:
    """Trying to write the detect_stream method as a better implimented class."""
    def __init__(self, weights='best.pt', source='http://ironacer.local:8000/stream.mjpg', imgsz=(1280, 1280), conf_thres=0.25, motion_detection_only=False):
        if motion_detection_only is False:
            self.weights = weights
            self.source = str(source)
            self.imgsz = imgsz  # inference size (height, width)
            self.conf_thres = conf_thres  # confidence threshold
            self.iou_thres = 0.45  # NMS IOU threshold
            self.max_det = 1000  # maximum detections per image
            self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            self.device = select_device(self.device)
            self.classes = None  # filter by class: --class 0, or --class 0 2 3
            self.agnostic_nms = False  # class-agnostic NMS
            self.nosave = False  # do not save images/videos
            self.augment = False  # augmented inference
            self.visualize = False  # visualize features
            self.line_thickness = 3  # bounding box thickness (pixels)
            self.hide_labels = False  # hide labels
            self.hide_conf = False  # hide confidences
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
        # For motion detector:
        self.prev_frame = None
        self.motion_list = [None, None]

    def stream(self):
        """Return raw image from the stream"""
        self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        for path, im, im0s, vid_cap, s in self.dataset:
            # Crop 2592x1944 image to 1280x1280.
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            yield path, im, im0s, vid_cap, s
            # for batch in self.inference(path, im, im0s, vid_cap, s):
            #     yield batch

        # If you get to this point, the stream has been dropped.
        raise AssertionError('Stream cannot be connected to.')

    def motion_detector(self, frame, motion_thresh=500):
        """
        Saves the current frame if there is significant enough movement from the previous frame.
        :param motion_thresh:
        :param frame:
        :return:
        """
        # Based on Webcam Motion Detector from https://www.geeksforgeeks.org/webcam-motion-detector-python/
        image_path = None
        if frame is None:
            return image_path

        motion = 0  # 0 = no motion, 1 = yes motion.

        og_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting color image to gray_scale image
        frame = cv2.GaussianBlur(frame, (21, 21), 0)

        if self.prev_frame is None:  # Init first frame to gray background.
            self.prev_frame = frame
            return image_path

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

            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            bounding_boxes.append(f'{x} {y} {w} {h}')

        if motion == 1:  # Save the image.
            ext_num = len([i for i in os.listdir('motion_detected/image/') if i.endswith(".jpg")]) + 1
            image_path = str(f'motion_detected/image/result-{ext_num}.jpg')
            cv2.imwrite(image_path, og_frame)  # Write image
            label_path = str(f'motion_detected/label/result-{ext_num}.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(bounding_boxes))

        self.prev_frame = frame
        return image_path

    @torch.no_grad()
    def inference(self, path, im, im0s, vid_cap, s):
        # path, im, im0s, vid_cap, s = self.stream_frame()
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        # Process predictions
        im0 = im0s[0].copy()
        pred = pred[0]
        isSquirrel = False
        xyxy = False
        confidence = False

        if len(pred):  # If found a squirrel, this is triggered.
            isSquirrel = True
            # det = tensor list of xmin, ymin, xmax, ymax, confidence, class number
            # Rescale boxes from img_size to im0 size, basically normalises it.
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(pred):  # the *xyxy is to take the first 4 items as the coords.
                confidence = conf.item()
                xyxy = [i.item() for i in xyxy]  # Convert from [tensor(x), ..] to [x, ..]
                self.save_train_data(im0, xyxy)

        vid_path = self.save_labeled(pred, im0)

        return isSquirrel, xyxy, confidence, vid_path

    def save_labeled(self, det, im0):
        """Should just need a frame and det."""
        save_dir = 'results/'

        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):  # There is a squirrel.
            self.number_of_frames_without_squirrel = 10
            for *xyxy, conf, cls in reversed(det):
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
            if len(det) or self.number_of_frames_without_squirrel > 0:  # record video
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

    def save_train_data(self, im0, coordinates):
        """Takes the image and corrdinates of the box and save them to training_wheels for future training.
        :return nothing. """
        # Write image and box to training_wheels for future training data.
        ext_num = len([i for i in os.listdir('training_wheels/images') if i.endswith(".jpg")]) + 1
        image_path = str(f'training_wheels/images/result-{ext_num}.jpg')
        labels_path = str(f'training_wheels/labels/result-{ext_num}.txt')
        cv2.imwrite(image_path, im0)  # Write image
        with open(labels_path, 'w') as f:  # Convert coordinates and save as txt file.
            # class (0 for squirrel, x_center y_center width height from top right of image and normalised to be 0-1.
            xmin, ymin, xmax, ymax = coordinates
            im_width, im_height = im0.shape[1], im0.shape[0]
            x_center = (ymin + ((ymax - ymin) / 2)) / im_width
            y_center = (xmin + ((xmax - xmin) / 2)) / im_height
            width = (xmax - xmin) / im_width
            height = (ymax - ymin) / im_height
            f.write(f'0 {str(x_center)} {str(y_center)} {str(width)} {str(height)}')


if __name__ == '__main__':
    motion_detect_img_dir(start_number=1000)
    # d = StreamDetector(motion_detection_only=True)
    # vid = cv2.VideoCapture('results/result-114.mp4')
    # while True:
    #     i, frame = vid.read()
    #     if frame is None:
    #         break
    #     print(d.motion_detector(frame, motion_thresh=1000))

    # d = StreamDetector(weights='best.pt')
    # for path, im, im0s, vid_cap, s in d.stream():
    #     isSquirrel, coords, confidence, vid_path = d.inference(path, im, im0s, vid_cap, s)
        # print(isSquirrel, coords, confidence, vid_path)

    # # For detect_stream
    # for i in detect_stream(source='http://ironacer.local:8000/stream.mjpg'):
    #     isSquirrel, coords, confidence, vid_path = i
    #     print(isSquirrel, coords, confidence, vid_path)
