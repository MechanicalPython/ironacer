"""
Mostly to hold random methods and classes.
"""


def label_by_total_motion(path):
    import os
    import cv2

    if not os.path.exists(f'{path}motion_ordered_images'):
        os.mkdir(f'{path}motion_ordered_images')

    for image in [f for f in os.listdir(f'{path}image/') if f.endswith('.jpg')]:
        if 'Motion' not in image:
            continue
        serial_number = image.replace('.jpg', '')
        print(serial_number)
        frame = cv2.imread(f'{path}image/{serial_number}.jpg')
        labels = open(f'{path}label/{serial_number}.txt', 'r').read().strip().split('\n')
        total_motion = 0
        for label in labels:
            x, y, w, h, amount_of_motion = label.split(' ')
            x, y, w, h, amount_of_motion = int(x), int(y), int(w), int(h), float(amount_of_motion)
            total_motion += amount_of_motion
        cv2.imwrite(f'{path}motion_ordered_images/{total_motion}-{serial_number}.jpg', frame)


def motion_detect_img_dir(path='detected/', detect_region=['0', '300', '1280', '800', 'Detect']):
    """Saves labeled images to new directory to analyse easily."""
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
    # motion_detect_img_dir(path='/Users/matt/detected/')
    label_by_total_motion('/Users/matt/detected/')
