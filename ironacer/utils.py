import os
import cv2
from ironacer import DETECTION_REGION, ROOT, MOTION_THRESH
import datetime


def make_video(images_dir, video_path, fps):
    """Should just need a frame and det.
    return None when the video is not ready. Return video path when ready to send out.

    Inputs: isSquirrel, xyxy, confidence, cls,
    """
    images = os.listdir(images_dir)
    images = [f'{images_dir}{i}' for i in images if i.endswith('jpg')]

    w, h, colours = cv2.imread(images[0]).shape
    vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for frame in images:
        vid_writer.write(cv2.imread(frame))

    vid_writer.release()  # release previous video writer

    return video_path


def save_frame(frame, xyxyl, origin):
    """Saves the inputted frame and label in ironacer/detected/image and ironacer/detected/label.
    label = x, y, x, y, label.
    xyxyl = [[x, y, x, y, l], ..]

    Can convert the yolo [[xyxy, confidence, cls], ..] if type is yolo.
    """
    if not os.path.exists(f'{ROOT}/detected/'):
        os.mkdir(f'{ROOT}/detected/')
    if not os.path.exists(f'{ROOT}/detected/image'):
        os.mkdir(f'{ROOT}/detected/image')
    if not os.path.exists(f'{ROOT}/detected/label'):
        os.mkdir(f'{ROOT}/detected/label')

    if origin == 'Yolo':
        labels = []  # Convert yolo results into cv2 labels.
        for result in xyxyl:
            xyxy, conf, cls = result  # xyxy is list of 4 items.
            xyxy.append(conf)  # add conf to xyxy to save it.
            labels.append(xyxy)
        xyxyl = labels

    t = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f'))
    image_path = f'{ROOT}/detected/image/{origin}_result-{t}.jpg'
    cv2.imwrite(image_path, frame)  # Write image
    label_path = f'{ROOT}/detected/label/{origin}_result-{t}.txt'

    label = ''
    for box in xyxyl:
        box = [str(i) for i in box]
        label = f'{label}{" ".join(box)}\n'

    with open(label_path, 'w') as f:
        f.write(label)

    return image_path, label_path


def add_label_to_frame(frame, xyxyl, yolo=False):
    """
    xyxyl = [[x, y, x, y, label], ] top left, bottom right.

    If using on DETECTION_REGION put it inside a list.
    """
    if yolo:
        labels = []  # Convert yolo results into cv2 labels.
        for result in xyxyl:
            xyxy, conf, cls = result  # xyxy is list of 4 items.
            xyxy.append(conf)  # add conf to xyxy to save it.
            labels.append(xyxy)
        xyxyl = labels
    for label in xyxyl:
        if None in label:
            continue
        if len(label) == 4:
            label.append(' ')
        x, y, x2, y2, amount_of_motion = label
        x, y, x2, y2, amount_of_motion = int(x), int(y), int(x2), int(y2), str(amount_of_motion)
        # making green rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, amount_of_motion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame


def show_frame(frame, rects=None):
    """

    :param frame:
    :param rects: list of [x, y, w, h, label] to put up labels.
    :return:
    """
    if rects is not None:
        for rect in rects:
            x, y, w, h, label = rect
            x, y, w, h, label = int(x), int(y), int(w), int(h), str(label)
            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Motion Box", frame)
    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        return False


def demonstrate_motion_detection():
    #  Demonstrate motion detection on macbook webcam.
    from stream import LoadCamera
    from motion_detection import MotionDetection
    motion_detector = MotionDetection(detection_region=DETECTION_REGION, motion_thresh=MOTION_THRESH)

    with LoadCamera() as stream:
        for frame in stream:
            is_motion, results = motion_detector.detect(frame)  # results = [[x, y, x, y, motion],.. ]
            if results is None:
                continue
            frame = add_label_to_frame(frame, results)
            frame = add_label_to_frame(frame, [DETECTION_REGION])  # Append to add the label.
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def demonstrate_yolo():
    #  Demonstrate motion detection on macbook webcam.
    from stream import LoadCamera
    from find import Detector
    yolo = Detector(weights='../best.pt')

    with LoadCamera() as stream:
        for frame in stream:
            is_squirrel, results = yolo.inference(frame)  # results = [[x, y, x, y, motion],.. ]
            if is_squirrel:

                frame = add_label_to_frame(frame, results, yolo=True)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def list_images_and_labels(path):
    """Return read image and labels from detected directory."""
    images = os.listdir(f'{path}image/')
    images.sort()
    for image in [f for f in images if f.endswith('.jpg')]:
        if 'Motion' not in image:
            continue
        serial_number = image.replace('.jpg', '')
        print(serial_number)
        # Reading frame(image) from video
        frame = cv2.imread(f'{path}image/{serial_number}.jpg')
        labels = open(f'{path}label/{serial_number}.txt', 'r').read().strip().split('\n')
        yield frame, labels, serial_number


def motion_detect_img_dir(path='detected/'):
    """Saves labeled images to new directory to analyse easily."""
    if not os.path.exists(f'{path}labeled_images'):
        os.mkdir(f'{path}labeled_images')

    for frame, labels, serial_number in list_images_and_labels(path):
        for label in labels:
            x, y, w, h, amount_of_motion = label.split(' ')
            x, y, w, h, amount_of_motion = int(x), int(y), int(w), int(h), str(amount_of_motion)
            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 3)
            cv2.putText(frame, amount_of_motion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imwrite(f'{path}labeled_images/{serial_number}.jpg', frame)


if __name__ == '__main__':
    demonstrate_yolo()
    # make_video('../Ironacer.v1-batch-1.yolov5pytorch/test/images/', 'test.mp4', 6)

    # motion_detect_img_dir(path='/Users/matt/detected/')
