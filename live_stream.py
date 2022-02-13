"""
Stream the camera for focusing and adjustment purposes.
"""

import logging
import socketserver
from http import server

from stream import LoadWebcam
import cv2
import argparse


PAGE = """\
<html>
<head>
<title>picamera MJPEG streaming demo</title>
</head>
<body>
<h1>PiCamera MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="640" height="480" />
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
                with LoadWebcam(capture_size=capture, output_img_size=output) as stream:
                    for frame in stream:
                        frame = cv2.imencode('.jpg', frame)[1].tobytes()

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
    parser.add_argument('--capture', type=str, default="1080,1080")
    parser.add_argument('--crop', type=str, default="1080,1080")
    return parser.parse_args()



if __name__ == '__main__':
    # http://ironacer.local:8000/stream.mjpg
    opt = arg_parse()
    capture = [int(i) for i in opt.capture.split(',')]
    output = [int(i) for i in opt.crop.split(',')]
    print(capture, output)
    # Global var as I CBA to pass them properly.

    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
