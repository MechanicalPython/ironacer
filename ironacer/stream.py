"""
Stream raw cv2 video, as an array, that other aspects of the program can plug into.

"""
import time

import cv2


class LoadCamera:
    """
    # A class to load a camera as context manager.
    Uses cv2 VideoCapture(0)

    # Usage
    with LoadCamera(output_img_size=(self.imgsz, self.imgsz)) as stream:
        for frame in stream:
            do something with the frame.

    # Notes
        hq camera - 4056 x 3040 pixels max resolution.

    # Credit
    Taken and modified from yolov5/utils LoadWebcam.
    """
    def __init__(self, resolution=(1280, 1280)):
        self.capture_size = resolution
        self._check_resolution(resolution)  # Errors out if not correct sizes.

        self.reset_time = time.time()
        self.reset_freq = 60 * 15  # Reset the camera every 15 mins to prevent the exposure problem.
        self.cap = None

    def set_camera(self):
        self.cap = cv2.VideoCapture(0)   # For pi0 - VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 is automatic exposure. 0.75 is manual control.
        self.cap.read()  # Clear buffer
        time.sleep(1)

    def __enter__(self):
        self.set_camera()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        if time.time() - self.reset_time > self.reset_freq:
            self.cap.release()
            self.set_camera()
            self.reset_time = time.time()
        # Read frame
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.25)
        ret_val, img = self.cap.read()
        if img is None:
            print(f'Frame is None. {self.get_all_settings()}')
            return None
        return img

    @staticmethod
    def _digital_crop(frame, x1, y1, x2, y2):
        """Digital crop. x1 and y1 is the top left of the image. x2 and y2 is the bottom right."""
        frame = frame[y1:y2, x1:x2]  # y1:y2, x1:x2 where x1y1 it top left and x2y2 is bottom right.
        return frame

    @staticmethod
    def _check_resolution(resolution):
        """The camera's block size is 32x16 so any image data provided to a renderer must have a width which is a
        multiple of 32, and a height which is a multiple of 16."""
        if resolution[0] % 32 != 0:
            raise Exception('Width must be multiple of 32')
        if resolution[1] % 16 != 0:
            raise Exception('Height must be multiple of 16')

    def get_all_settings(self):
        return f"""
        CAP_PROP_MODE:  {str(self.cap.get(cv2.CAP_PROP_MODE))}
        CAP_PROP_FPS:  {str(self.cap.get(cv2.CAP_PROP_FPS))}
        CAP_PROP_CONTRAST:  {str(self.cap.get(cv2.CAP_PROP_CONTRAST))}
        CAP_PROP_GAIN:  {str(self.cap.get(cv2.CAP_PROP_GAIN))}
        CAP_PROP_FRAME_WIDTH:  {str(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}
        CAP_PROP_FRAME_HEIGHT:  {str(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))}
        CAP_PROP_POS_FRAMES:  {str(self.cap.get(cv2.CAP_PROP_POS_FRAMES))}
        CAP_PROP_EXPOSURE:  {str(self.cap.get(cv2.CAP_PROP_EXPOSURE))}
        CAP_PROP_AUTO_EXPOSURE:  {str(self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))}"""

