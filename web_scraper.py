#! /usr/local/bin/python3.7
"""
Scrape the data from flickr. Must be run with python 3.7 for 3.9 deprecation reasons.
"""

from flickrapi import FlickrAPI
import requests
import os
import time
from PIL import Image


class FlickrDownload:
    def __init__(self, image_tags, max_downloads=2000):
        self.KEY = open("KEY", 'r').read()
        self.SECRET = open("SECRET", 'r').read()
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

                with open(image_path, 'wb') as outfile:
                    outfile.write(response.content)

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
    for image in os.listdir(dir):
        if image.endswith('JPEG'):
            print(f'{dir}/{image}')
            im = Image.open(f'{dir}/{image}')
            im.thumbnail(max_size)
            im.save(f'{dir}/{image}')



if __name__ == '__main__':
    FlickrDownload(['grey squirrel']).main()
    resize_images('data/grey squirrel')

