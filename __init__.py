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


def motion_detect_img_dir(path='detected/', detect_region=['0', '350', '1280', '600', 'Detect']):
    """Saves labeld images to new directory to analyse easily."""
    import cv2

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
