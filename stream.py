"""
Stream raw cv2 video, as an array, that other aspects of the program can plug into.

"""

import cv2
import time


def show_frame(frame, rects=None):
    """

    :param frame:
    :param rect: list of [x, y, w, h, label] to put up labels.
    :return:
    """
    if rects is not None:
        for rect in rects:
            x, y, w, h, label = rect
            x, y, w, h, label = int(x), int(y), int(w), int(h), str(label)
            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Motion Box", frame)
    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        return False


class LoadWebcam:
    """
    Taken and modified from yolov5/utils LoadWebcam.
    Returns just the image, the augmentation needed for inference is done by find.py.
    """

    def __init__(self, pipe='0', capture_size=(3280, 2464), output_img_size=1280, stride=32, on_mac=True):
        self.capture_size = capture_size
        self.output_img_size = output_img_size

        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.on_mac = on_mac
        # self.set_camera()
        self.reset_freq = 5*60  # Frequency to reset the camera (in seconds).
        self.t = time.time()

    def set_camera(self):
        if self.on_mac:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        time.sleep(0.5)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        width, height = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        x1 = int((width / 2) - (self.output_img_size / 2))
        y1 = int((height / 2) - (self.output_img_size / 2))

        x2 = int((width / 2) + (self.output_img_size / 2))
        y2 = int((height / 2) + (self.output_img_size / 2))

        self.crop_xyxy = [x1, y1, x2, y2]

    def __enter__(self):
        print('start camera')
        self.set_camera()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.cap.release()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        # Read frame
        ret_val, img = self.cap.read()
        if img is None:
            return None

        # Crop image to correct size.
        x1, y1, x2, y2 = self.crop_xyxy
        img = img[y1:y2, x1:x2]  # y1:y2, x1:x2 where x1y1 it top left and x2y2 is bottom right.

        if time.time() - self.t > self.reset_freq:
            self.reset_camera()
            self.t = time.time()
        return img

    def reset_camera(self):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)


# max - 3280 × 2464 pixels
# 1-15 fps - 2592 x 1944


if __name__ == '__main__':
    import telegram_bot
    bot = telegram_bot.TelegramBot()
    with LoadWebcam() as stream:
        for img in stream:
            show_frame(img)
            img = cv2.imencode('.jpg', img)[1].tobytes()  # cv2.imencode gives True, array, dtype
            # print(type(img))
            bot.send_photo(img)
            # quit()
            cv2.enco




