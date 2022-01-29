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


def motion_detect_img_dir(path='detected/', start_number=1):
    """Shows images and where the motion was detected."""
    import os
    import cv2
    import time
    # Loop to treat stack of image as video

    images_dir = f'{path}image/'
    for image in [f for f in os.listdir(images_dir) if f.endswith('.jpg')]:
        if 'Motion' not in image:
            continue
        serial_number = image.split('-')[1].replace('.jpg', '')
        # Reading frame(image) from video
        frame = cv2.imread(f'{path}image/Motion_result-{serial_number}.jpg')
        labels = open(f'{path}label/Motion_result-{serial_number}.txt', 'r').read().split('\n')
        for label in labels:
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
    motion_detect_img_dir(path='/Users/matt/Downloads/detected/', start_number=1)
    # motion_detected_squirrel_organiser("164 - 171, 253 - 262, 397 - 415, 980 - 999, 1919 - 1929")
