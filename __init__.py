class FlickrDownload:
    from flickrapi import FlickrAPI
    import requests
    import os
    import time

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
