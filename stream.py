import logging
import socketserver
from http import server
import sys

import cv2

PAGE = """\
<html>
<head>
<title>Ironacer Live View</title>
</head>
<body>
<h1>Ironacer Live View</h1>
<img src="stream.mjpg" width="1280" height="1280" />
</body>
</html>
"""


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    frame = next(self.crop_video(cap, crop_xywh))
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

    @staticmethod
    def crop_video(cap, crop_xywh):
        if not cap.isOpened():
            raise Exception("Could not open video device")
        x, y, w, h = crop_xywh
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = frame[y:y + h, x:x + w]
                _, JPEG = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                yield JPEG.tobytes()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# max - 3280 × 2464 pixels
# 1-15 fps - 2592 x 1944

if __name__ == '__main__':
    width, height, size = sys.argv
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    x = (width - size) / 2
    y = (height - size) / 2
    crop_xywh = (x, y, size, size)

    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
