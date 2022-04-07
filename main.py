"""
Notes:
    Not going to try and account for false negatives. It's too complex. If it turns out that the AI is missing squirrels
    then I'll do something about it.


"""
import argparse
import datetime
import os
import time
import threading
import suntime

from ironacer import strike, telegram_bot, stream, find, motion_detection, utils
from ironacer import ROOT, DETECTION_REGION, YOLO_WEIGHTS, IMGSZ, MOTION_THRESH


# todo
#  run telegram, inference, and motion detection on separate threads to speed it up.
#  Too many photos - pull_motion...sh could go back to downloading the individual photos, not the zip.


class IronAcer:
    """
    Motion detection just used to trigger yolo or not.

    Only saves yolo detected images of squirrels plus previous image and the next 5 images.

    Runs forever with a service: https://www.tomshardware.com/how-to/run-long-running-scripts-raspberry-pi

    """

    def __init__(self,
                 surveillance_mode=False,  # Don't run the strike functions.
                 gather_data=True):  # Keep as the systemctl service expects it.
        self.surveillance_mode = surveillance_mode

        self.yolo = find.Detector(YOLO_WEIGHTS, (IMGSZ, IMGSZ))

        self.motion_detector = motion_detection.MotionDetection(
            detection_region=DETECTION_REGION, motion_thresh=MOTION_THRESH)
        self.claymore = strike.Claymore()

        self.bot = telegram_bot.TelegramBot()
        self.bot.claymore = self.claymore

        self.sun = suntime.Sun(51.5, -0.1)  # London lat long.

    def is_daytime(self):
        sunrise = self.sun.get_sunrise_time().replace(tzinfo=None)
        sunset = self.sun.get_local_sunset_time().replace(tzinfo=None)
        return sunrise < datetime.datetime.now() < sunset

    def cpu_temp(self):
        """
        Continuous thread for checking the cpu temp and messaging telegram.
        """
        while True:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                temp = int(f.read().strip()) / 1000
                if temp > 80:
                    self.bot.send_message(f'Warning: CPU temperature is {temp}')
            time.sleep(5)

    def main(self):
        """
        Runs yolo inference on frames with enough motion detection, to save power and reduce constant load on the
        pi.
        """
        temp_thread = threading.Thread(target=self.cpu_temp, daemon=True)
        temp_thread.start()
        telegram_thread = threading.Thread(target=self.bot.main, daemon=True)
        telegram_thread.start()

        squirrel_cooldown = -1
        with stream.LoadCamera(resolution=(IMGSZ, IMGSZ)) as frames:
            while True:
                # If it is nighttime, just go to sleep like you should.
                if not self.is_daytime():
                    time.sleep(60)
                    continue

                frames.__next__()  # Clear buffer twice to fix the black image at start up problem.
                frames.__next__()
                self.bot.send_photo(utils.add_label_to_frame(frames.__next__(), [DETECTION_REGION]))  # Telegram start up msg.
                for frame in frames:
                    self.bot.latest_frame = frame

                    is_motion, motion_detection_result = self.motion_detector.detect(frame)
                    if is_motion:
                        is_squirrel, inference_result = self.yolo.inference(frame)
                        if is_squirrel:
                            squirrel_cooldown = 10
                            # vid_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (IMGSZ, IMGSZ))
                            # self.claymore.start()
                        else:
                            squirrel_cooldown -= 1
                    else:
                        squirrel_cooldown -= 1

                    if squirrel_cooldown > 0:
                        utils.save_frame(frame, inference_result, 'Yolo')
                        # vid_writer.write(utils.add_label_to_frame(frame, inference_result, 'Yolo'))
                    if squirrel_cooldown == 0:
                        # self.claymore.stop()
                        # vid_writer.release()
                        # with open('temp.mp4') as f:
                        #     self.bot.send_video(f)
                        # os.remove('temp.mp4')
                        squirrel_cooldown = -1
                    if squirrel_cooldown < -3:
                        squirrel_cooldown = -1

                    if not self.is_daytime():
                        self.bot.send_message(f"{len(os.listdir(f'{ROOT}/detected/image/'))} images currently saved")
                        break


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surveillance_mode', action='store_true', help='Flag to not fire water.')
    parser.add_argument('--gather_data', action='store_true', help='Only gather data with motion detection')
    return parser.parse_args()


if __name__ == '__main__':
    opt = arg_parse()
    IA = IronAcer(**vars(opt))
    # IA.bot.chat_id = 1706759043  # Change it to private chat for testing.
    IA.main()
