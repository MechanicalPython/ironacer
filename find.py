
"""
FIND.py - find the exact location of the squirrel in the image in 3d space.
"""

import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import time

from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


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


# Deprecated - used for static images.
class ImageDetector:
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

        yield isSquirrel, xyxy, confidence, vid_path

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
                  nosave=False,  # do not save images/videos
                  augment=False,  # augmented inference
                  visualize=False,  # visualize features
                  line_thickness=3,  # bounding box thickness (pixels)
                  hide_labels=False,  # hide labels
                  hide_conf=False,  # hide confidences
                  half=False,  # use FP16 half-precision inference
                  dnn=False,  # use OpenCV DNN for ONNX inference
                  ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    save_dir = 'results/'
    vid_num = len([i for i in os.listdir(save_dir) if i.endswith(".mp4")]) + 1
    save_path = str(f'{save_dir}result-{vid_num}.mp4')
    vid_writer = None
    # Tracking when to start a new video
    number_of_frames_without_squirrel = 0

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    # dt, seen = [0.0, 0.0, 0.0], 0
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
        # pred gives [tensor]
        # Process predictions
        for i, det in enumerate(pred):  # per image - i think, only useful if you pass it multiple images.
            # det gives tensor.
            # seen += 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):  # If found a squirrel, this is triggered.
                isSquirrel = True
                number_of_frames_without_squirrel = 10  # How many frames in a row can be false before resetting the vid
                # det = tensor list of xmin, ymin, xmax, ymax, confidence, class number
                # Rescale boxes from img_size to im0 size, basically normalises it.
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                coordinates = det[:, :4]
                confidence = det[:, 5]
                # Write image and box to training_wheels for future training data.
                ext_num = len([i for i in os.listdir(save_dir) if i.endswith(".jpg")]) + 1
                image_path = str(f'training_wheels/images/result-{ext_num}.jpg')
                labels_path = str(f'training_wheels/labels/result-{ext_num}.txt')
                cv2.imwrite(image_path, im0)
                with open(labels_path, 'w') as f:  # class (0 for squirrel, x_center y_center width height
                    # from top right of image and normalised to be 0-1.
                    xmin, ymin, xmax, ymax = coordinates[0]
                    im_width, im_height = im0.shape[1], im0.shape[0]
                    x_center = (ymin + ((ymax - ymin) / 2)) / im_width
                    y_center = (xmin + ((xmax - xmin) / 2)) / im_height
                    width = (xmax - xmin) / im_width
                    height = (ymax - ymin) / im_height
                    f.write(f'0 {str(x_center)} {str(y_center)} {str(width)} {str(height)}')

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add box to image.
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            else:
                if number_of_frames_without_squirrel > 0:
                    number_of_frames_without_squirrel -= 1

                isSquirrel = False
                coordinates = False
                confidence = False

            vid_path = False
            if save_img:
                im0 = annotator.result()
                if isSquirrel or number_of_frames_without_squirrel > 0:  # Record video
                    if os.path.exists(save_path):  # Write to current file
                        vid_writer.write(im0)  # Despite being inited above with None, this can't be reached unless
                        # the else part has run, therefore don't worry about it.
                    else:  # Start a new file
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        fps, w, h = 6, im0.shape[1], im0.shape[0]
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
                else:
                    vid_num = len([i for i in os.listdir(save_dir) if i.endswith(".mp4")]) + 1
                    save_path = str(f'{save_dir}result-{vid_num}.mp4')
                    vid_path = str(f'{save_dir}result-{vid_num - 1}.mp4')  # Now okay to send the just saved video out.

            yield isSquirrel, coordinates, confidence, vid_path

    # If you get to this point, the stream has been dropped.
    raise AssertionError('Stream cannot be connected to')


def motion_detect_img_dir(path='motion_detected/'):
    # Assigning our static_back to None
    prev_frame = None

    # Infinite while loop to treat stack of image as video
    for i in range(1, len([i for i in os.listdir(f'{path}image/') if i.endswith(".jpg")]) + 1):
        # Reading frame(image) from video
        frame = cv2.imread(f'{path}image/result-{i}.jpg')
        labels = open(f'{path}label/result-{i}.txt', 'r').read().split('\n')
        for label in labels:
            x, y, w, h = [int(n) for n in label.split(' ')]
            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # cv2.imshow("Gray Frame", gray)
        # cv2.imshow("Difference Frame", diff_frame)
        # cv2.imshow("Threshold Frame", thresh_frame)

        cv2.imshow("Motion Box", frame)
        time.sleep(0.2)
        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            break

    # Destroying all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    motion_detect_img_dir()
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
