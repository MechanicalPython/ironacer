"""
Stream the camera for focusing and adjustment purposes.
"""

import logging
import socketserver
from http import server

from stream import LoadWebcam
import cv2
import argparse
import os


PAGE = """\
<html>
<head>
<title>picamera MJPEG streaming demo</title>
</head>
<body>
<h1>PiCamera MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="640" height="640" />
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
                with LoadWebcam(capture_size=capture) as stream:
                    for frame in stream:
                        frame = cv2.imencode('.jpg', frame)[1].tobytes()
                        with open('exposure.txt') as f:
                            exposure = float(f.read().strip())
                            if exposure == 0.25:
                                stream.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, exposure)
                            elif exposure == 0.75:
                                stream.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, exposure)
                            else:
                                stream.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                            print(exposure, stream.get_all_settings())
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


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=str, default="1280,1280")
    return parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists('exposure.txt'):
        with open('exposure.txt', 'w') as f:
            f.write('0.25')
    # http://ironacer.local:8000/stream.mjpg
    opt = arg_parse()
    capture = [int(i) for i in opt.capture.split(',')]
    print(capture)
    # Global var as I CBA to pass them properly.
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        pass

