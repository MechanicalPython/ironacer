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


def motion_detect_img_dir(path='detected/', start_number=1, detect_region=['0', '350', '1280', '400', 'Detect']):
    """Shows images and where the motion was detected."""
    import os
    import cv2
    import time
    # Loop to treat stack of image as video

    images_dir = f'{path}image/'
    os.listdir(images_dir)
    while True:
        for image in [f for f in os.listdir(images_dir) if f.endswith('.jpg')]:
            if 'Motion' not in image:
                continue
            serial_number = image.replace('.jpg', '')
            # Reading frame(image) from video
            frame = cv2.imread(f'{path}image/{serial_number}.jpg')
            labels = open(f'{path}label/{serial_number}.txt', 'r').read().strip().split('\n')
            labels.append(' '.join(detect_region))
            for label in labels:
                print(len(label))
                x, y, w, h, amount_of_motion = label.split(' ')
                x, y, w, h, amount_of_motion = int(x), int(y), int(w), int(h), str(amount_of_motion)
                # making green rectangle around the moving object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, amount_of_motion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            cv2.imshow("Motion Box", frame)
            time.sleep(1)
            key = cv2.waitKey(1)
            # if q entered whole process will stop
            if key == ord('q'):
                print(f'Last image shown: {image}')
                break
            elif key == ord('t'):
                print(f'Image: {image}')

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
    motion_detect_img_dir(path='/Users/matt/Downloads/Archive/', start_number=1)
    # motion_detected_squirrel_organiser("164 - 171, 253 - 262, 397 - 415, 980 - 999, 1919 - 1929")
