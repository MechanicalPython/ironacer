"""
Mostly to hold random methods and classes.
"""

import cv2

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


def motion_detect_img_dir(path='detected/', detect_region=['0', '350', '1280', '600', 'Detect']):
    """Saves labeld images to new directory to analyse easily."""
    import cv2
    import os

    if not os.path.exists(f'{path}labeled_images'):
        os.mkdir(f'{path}labeled_images')

    for image in [f for f in os.listdir(f'{path}image/') if f.endswith('.jpg')]:
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
