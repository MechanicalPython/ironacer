#! /usr/local/bin/python3.7

"""
File to contain the one-shot methods that are used.
Flickr download and resize images are just to download and prep the photos for bounding boxes for the darknet training.
"""
from flickrapi import FlickrAPI
import requests
import os
import time
from PIL import Image


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
        flickr = FlickrAPI(self.KEY, self.SECRET)
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

    @staticmethod
    def download_images(urls, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        for url in urls:
            image_name = url.split("/")[-1]
            image_path = os.path.join(path, image_name)

            if not os.path.isfile(image_path):  # ignore if already downloaded
                response = requests.get(url, stream=True)
                try:
                    with open(image_path, 'wb') as outfile:
                        outfile.write(response.content)
                except requests.exceptions.ChunkedEncodingError as e:
                    print(f'{e} for {url}')

    def main(self):
        start = time.time()
        for tag in self.image_tags:
            print('Getting urls for ', tag)
            urls = self.get_urls(tag)

            print(f'Downloading {len(urls)} images for {tag}')
            path = os.path.join('data', tag)
            self.download_images(urls, path)

        print(f'Took {round(time.time() - start, 2)} seconds')


def resize_images(dir, max_size=(1080, 1080)):
    """
    Used to resize a directory of images to a max dimentions
    :param dir: Directory of photos.
    :param max_size:
    :return:
    """
    if not dir.endswith('/'):
        dir = f'{dir}/'
    dir = os.path.expanduser(dir)
    endings = ['jpg', 'jpeg', 'png']
    for image in os.listdir(dir):
        if any(image.endswith(end.lower()) for end in endings):
            print(f'{dir}{image}')
            im = Image.open(f'{dir}{image}')
            im.thumbnail(max_size)
            im.save(f'{dir}{image}')


if __name__ == '__main__':
    # FlickrDownload(['heron']).main()
    resize_images('data/heron/')