import os
import cv2
from ironacer import DETECTION_REGION, ROOT, MOTION_THRESH
import datetime


def save_results(frame, xyxyl, type):
    """Saves the inputted frame and label in ironacer/detected/image and ironacer/detected/label.
    label = x, y, x, y, label.
    xyxyl = [[x, y, x, y, l], ..]

    Can convert the yolo [[xyxy, confidence, cls], ..] if type is yolo.
    """
    if type == 'Yolo':
        labels = []  # Convert yolo results into cv2 labels.
        for result in xyxyl:
            xyxy, conf, cls = result  # xyxy is list of 4 items.
            xyxy.append(conf)  # add conf to xyxy to save it.
            labels.append(xyxy)
        xyxyl = labels

    t = str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f'))
    image_path = f'{ROOT}/detected/image/{type}_result-{t}.jpg'
    cv2.imwrite(image_path, frame)  # Write image
    label_path = f'{ROOT}/detected/label/{type}_result-{t}.txt'

    label = ''
    for box in xyxyl:
        box = [str(i) for i in box]
        label = f'{label}{" ".join(box)}\n'

    with open(label_path, 'w') as f:
        f.write(label)


def add_label_to_frame(frame, xyxyl):
    """
    xyxyl = [[x, y, x, y, label], ] top left, bottom right.
    """
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
            frame = add_label_to_frame(frame,  DETECTION_REGION.append('DETECT'))  # Append to add the label.
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def motion_detect_img_dir(path='detected/', detect_region=DETECTION_REGION.append('Detect')):
    """Saves labeled images to new directory to analyse easily."""
    if not os.path.exists(f'{path}labeled_images'):
        os.mkdir(f'{path}labeled_images')

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
        labels.append(' '.join(detect_region))
        for label in labels:
            x, y, w, h, amount_of_motion = label.split(' ')
            x, y, w, h, amount_of_motion = int(x), int(y), int(w), int(h), str(amount_of_motion)
            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 3)
            cv2.putText(frame, amount_of_motion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imwrite(f'{path}labeled_images/{serial_number}.jpg', frame)


if __name__ == '__main__':
    motion_detect_img_dir(path='/Users/matt/detected/')
    # label_by_total_motion('/Users/matt/detected/')
