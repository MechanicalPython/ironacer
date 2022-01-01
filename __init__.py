"""
Mostly to hold random methods and classes.
"""
import os


def next_free_path(path_pattern):
    """
    Method to save a file to a directory and increment the number on it.
    eg. save result-x.jpg to a directory and automatically increment x.
    From: https://stackoverflow.com/questions/17984809/how-do-i-create-an-incrementing-filename-in-python#17984925

    Finds the next free path in a sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':
    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
     :return:
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)
    return path_pattern % b


class FlickrDownload:
    """
    A basic class to download images from a flickr search. Uses flickr api so requires public and private keys to
    be saved in the ironacer directory.

    Use: FlickrDownload([image tag to be searches], max_downloads=2000).main()
    Files saved to working directory as
        data/tag1/tag_1.jpg
                 /tag_2.jpg
            /tag2/...

    NB: Must be run with python 3.7 for deprecation reasons.

    """
    # from flickrapi import FlickrAPI
    import requests
    import os
    import time

    def __init__(self, image_tags, max_downloads=2000):
        try:
            self.KEY = open("flickr_key", 'r').read()
            self.SECRET = open("flickr_secret", 'r').read()
        except FileNotFoundError as e:
            raise(e, 'Public or private flickr api key not found as: flickr_key and flickr_secret')
        self.sizes = ["url_o", "url_k", "url_h", "url_l", "url_c"]
        self.image_tags = image_tags
        self.max = max_downloads

    def get_photos(self, image_tag):
        extras = ','.join(self.sizes)
        flickr = self.FlickrAPI(self.KEY, self.SECRET)
        photos = flickr.walk(text=image_tag,
                             extras=extras,
                             privacy_filter=1,
                             per_page=50,
                             sort='relevance')
        return photos

    def get_url(self, photo):
        for i in range(len(self.sizes)):
            url = photo.get(self.sizes[i])
            if url:
                return url

    def get_urls(self, image_tag):
        photos = self.get_photos(image_tag)
        counter = 0
        urls = []

        for photo in photos:
            if counter < self.max:
                url = self.get_url(photo)
                if url:
                    urls.append(url)
                    counter += 1
            else:
                break
        return urls

    def download_images(self, urls, path):
        if not self.os.path.isdir(path):
            self.os.makedirs(path)

        for url in urls:
            image_name = url.split("/")[-1]
            image_path = self.os.path.join(path, image_name)

            if not self.os.path.isfile(image_path):  # ignore if already downloaded
                response = self.requests.get(url, stream=True)
                try:
                    with open(image_path, 'wb') as outfile:
                        outfile.write(response.content)
                except self.requests.exceptions.ChunkedEncodingError as e:
                    print(f'{e} for {url}')

    def main(self):
        start = self.time.time()
        for tag in self.image_tags:
            print('Getting urls for ', tag)
            urls = self.get_urls(tag)

            print(f'Downloading {len(urls)} images for {tag}')
            path = self.os.path.join('data', tag)
            self.download_images(urls, path)

        print(f'Took {round(self.time.time() - start, 2)} seconds')


class ImageDetector:
    """
    darknet_detect() box gives left, top, width, height. left and top gives the top left coordinate
    for the box. The width and height then allows you to work out the next 3 corners of the box.
    """
    import os
    import torch

    def __init__(self):
        self.model = self.torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
        self.model.cpu()
        self.model.conf = 0.1

    def get_latest_saved_image_path(self):
        """Return path to the latest saved image in runs/detect_from_image/exp*/.jpg"""
        root = 'runs/detect_from_image/'
        exps = self.os.listdir(root)
        exps.sort()
        exp = f'{root}{(exps[len(exps)-1])}/'
        images = self.os.listdir(exp)
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


# @torch.no_grad()
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


def motion_detect_img_dir(path='motion_detected/', start_number=1):
    import os
    import cv2
    import time
    # Loop to treat stack of image as video
    for i in range(start_number, len([i for i in os.listdir(f'{path}image/') if i.endswith(".jpg")]) + 1):
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
        time.sleep(0.1)
        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            print(f'Last image shown: {i}')
            break
        elif key == ord('t'):
            print(f'Image: {i}')

    # Destroying all the windows
    cv2.destroyAllWindows()


def motion_detected_squirrel_organiser(conf_phot_num):
    """
    Input range of file names from the motion detector and move and rename them to the confirmed_squirrel
    folder and delete the reviewed images.
    format as: 53 - 56, 98 - 103, 1057 - 1080, 1098 - 1099, 1116 - 1150, 1162 - 1165
    :return:
    """
    import os

    # Run through each file in order
    # Delete if not wanted, move is wanted.

    images = 'motion_detected/image/'
    labels = 'motion_detected/label/'
    confirmed = 'motion_detected/Confirmed_squirrel/'
    images_to_keep = []
    if conf_phot_num is not None:
        for r in conf_phot_num.split(', '):
            first, last = [int(x) for x in r.split(' - ')]
            for i in range(first, last + 1):
                images_to_keep.append(i)
    images_to_keep = [f'{images}result-{i}.jpg' for i in images_to_keep]
    all_images = [f'{images}{y}' for y in os.listdir(images) if y.endswith('.jpg')]
    all_labels = [f'{labels}{y}' for y in os.listdir(labels) if y.endswith('.txt')]
    for image, label in zip(all_images, all_labels):
        if os.path.exists(image):
            if image in images_to_keep:
                print('keep', image)
                suf = len([y for y in os.listdir(confirmed) if y.endswith('.jpg')]) + 1  # Number of images in conf dir
                os.rename(image, f'{confirmed}is_squirrel-{suf}.jpg')
                os.remove(label)
            else:
                print('delete', image)
                os.remove(image)
                os.remove(label)
        else:
            raise FileNotFoundError


if __name__ == '__main__':
    # motion_detect_img_dir(start_number=1)
    motion_detected_squirrel_organiser("164 - 171, 253 - 262, 397 - 415, 980 - 999, 1919 - 1929")


    # # For detect_stream
    # for i in detect_stream(source='http://ironacer.local:8000/stream.mjpg'):
    #     isSquirrel, coords, confidence, vid_path = i
    #     print(isSquirrel, coords, confidence, vid_path)

