
import html
import json
import os
import time
import traceback
import cv2
import subprocess
import configparser
import zipfile

from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext

from ironacer import ROOT, DETECTION_REGION
from ironacer import utils, strike

# todo - record the time before and after /fire.
#  temp alert spamming problem.
#  set detect region problem.


class TelegramBot:
    """
    Standard telegram bot

    # Bot Commands
    help
    /view: take a photo.
    /saved: get number of saved photos
    /set_detect x,y,x,y: set detection region
    /fire x: fire water for x seconds. Default 2 seconds.
    /update: update from github and reboot.
    /download: downloads all images in ironacer/detected as multiple 45MB zips.

    # Methods
    send_message - takes text.
    send_photo - takes photo as bytes.
    send_video - takes video as bytes.
    send_doc - takes file_path.
    Max file size is 50MB or 10MB photo.

    Around 75 photos is 50MB.

    # Notes
    Telegram bot either had to be run in a parallel thread to be used a real time 2-way interaction, or it can be used
    to passively send information at pre-determined times.
    """

    def __init__(self):
        self.token = open(f'{ROOT}/telegram_token', 'r').read()
        self.updater = Updater(self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.bot = self.updater.bot
        self.chat_id = -547385621
        self.latest_frame = None
        self.claymore = strike.Claymore()

    @staticmethod
    def detected_info():
        img_dir = f'{ROOT}/detected/image/'
        label_dir = f'{ROOT}/detected/label/'
        images_size = sum([os.path.getsize(f'{img_dir}{f}') for f in os.listdir(img_dir)])/1000000
        labels_size = sum([os.path.getsize(f'{label_dir}{f}') for f in os.listdir(label_dir)])/1000000
        total = round(images_size + labels_size, 2)
        return f"{total}MB total\n" \
               f"{len([i for i in os.listdir(img_dir) if 'yolo' in i.lower()])} yolo images and " \
               f"{len([i for i in os.listdir(img_dir) if 'motion' in i.lower()])} motion images"

    @staticmethod
    def help(update, context):
        update.message.reply_text('/view: take a photo.\n'
                                  '/saved: get number of saved photos\n'
                                  '/set_detect x,y,x,y: set detection region\n'
                                  '/fire x: fire water for x seconds\n'
                                  '/update: update from github and reboot.\n'
                                  '/download: downloads all images in ./detected as zips.\n'
                                  '/set_video: {true/false}: send video or not.'
                                  )

    def view(self, update, context):
        """Accepts self.latest_frame which is a cv2 np.array()"""
        frame = utils.add_label_to_frame(self.latest_frame, [DETECTION_REGION])
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        update.message.reply_photo(frame, timeout=300)

    def saved(self, update, context):
        update.message.reply_text(self.detected_info())

    def set_detect(self, update, context):
        # todo - how to update in real time?
        #  Git problem. local changes and then push to github?
        """Input is /set_detect x,y,x,y
        Not yet live.
        """
        args = update.message.text.split(' ')[1]
        parser = configparser.ConfigParser()
        parser.read(f'{ROOT}/settings.cfg')

        parser.set('Settings', 'DETECTION_REGION', args)
        with open(f'{ROOT}/settings.cfg', 'w') as configfile:
            parser.write(configfile)
        subprocess.Popen(["sudo", "systemctl", "restart", "ironacer"], stdout=subprocess.PIPE)

    def fire(self, update, context):
        """Fires water for a given amount of time. Default is 2 seconds."""
        args = update.message.text.split(' ')
        if len(args) == 1:
            duration = 2
        else:
            duration = int(args[1])
        self.claymore.start()
        time.sleep(duration)
        self.claymore.stop()

    def update(self, update, context):
        update.message.reply_text('Updating...')
        subprocess.Popen(["bash", "/home/pi/ironacer/updater.sh"], stdout=subprocess.PIPE)

    def download(self, update, context):
        self.zip_and_send()

    def set_video_setting(self, update, context):
        parser = configparser.ConfigParser()
        parser.read(f'{ROOT}/settings.cfg')

        args = update.message.text.split(' ')
        if len(args) == 2:
            if args[1].lower() == 'true':
                parser.set('Settings', 'SEND_VIDEO', args[1].lower())
                self.send_message(f'Send video set to true')
            elif args[1].lower() == 'false':
                parser.set('Settings', 'SEND_VIDEO', args[1].lower())
                self.send_message(f'Send video set to false')
            else:
                self.send_message(f'Invalid args. Only true or false are accepted. ')
            # Write changes.
            with open(f'{ROOT}/settings.cfg', 'w') as configfile:
                parser.write(configfile)
            self.send_message(f'Send video set to true')
        else:
            self.send_message(f'Invalid argument length. No changes made.')

    def zip_and_send(self, path=f'{ROOT}/detected/', max_zip_size=45):
        zf = zipfile.ZipFile(f"{ROOT}/detected.zip", 'w')

        total_size = 0  # Bytes. divide by 1000000 to get MB.
        for dirname, subdirs, files in os.walk(path):
            for filename in files:
                if not (filename.endswith('.jpg') or filename.endswith('.txt')):  # Filter out .DS_Store
                    continue

                file_size = os.path.getsize(f'{dirname}/{filename}') / 1000000  # Convert to MB.
                if file_size > max_zip_size:
                    self.bot.send_message(f'{filename} too large to send.')
                elif total_size + file_size > max_zip_size:
                    # Close zip file, send it, make a new one and continue.
                    zf.close()
                    self.send_doc(f"{ROOT}/detected.zip")
                    os.remove(f"{ROOT}/detected.zip")
                    zf = zipfile.ZipFile(f"{ROOT}/detected.zip", 'w')
                    total_size = 0

                else:
                    zf.write(f'{dirname}/{filename}')
                    total_size += file_size
                    os.remove(f'{dirname}/{filename}')
        # At the end, send the last zip file.
        zf.close()
        self.send_doc(f"{ROOT}/detected.zip")
        os.remove(f"{ROOT}/detected.zip")

    def send_message(self, text):
        self.bot.send_message(self.chat_id, text)

    def send_photo(self, frame):
        """
        Accepts cv2 frame array as input only.
        :param photo: cv2 frame np.array()
        """
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        self.bot.sendPhoto(self.chat_id, frame, timeout=300)

    def send_video(self, vid_bytes):
        self.bot.sendVideo(self.chat_id, vid_bytes, timeout=300)

    def send_doc(self, file_path):
        with open(file_path, 'rb') as f:
            self.bot.send_document(self.chat_id, f, timeout=300)

    def error(self, update: object, context: CallbackContext):
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = ''.join(tb_list)

        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f'An exception was raised while handling an update\n'
            f'<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}'
            '</pre>\n\n'
            f'<pre>context.chat_data = {html.escape(str(context.chat_data))}</pre>\n\n'
            f'<pre>context.user_data = {html.escape(str(context.user_data))}</pre>\n\n'
            f'<pre>{html.escape(tb_string)}</pre>'
        )
        # Finally, send the message
        context.bot.send_message(chat_id=self.chat_id, text=message, parse_mode=ParseMode.HTML)

    def main(self):
        self.dispatcher.add_handler(CommandHandler('help', self.help))
        self.dispatcher.add_handler(CommandHandler('view', self.view))
        self.dispatcher.add_handler(CommandHandler('saved', self.saved))
        self.dispatcher.add_handler(CommandHandler('fire', self.fire))
        self.dispatcher.add_handler(CommandHandler('update', self.update))
        self.dispatcher.add_handler(CommandHandler('download', self.download))
        self.dispatcher.add_handler(CommandHandler('set_video', self.set_video_setting))

        self.dispatcher.add_error_handler(self.error)

        self.updater.start_polling()
        # self.updater.idle()  # Comment out to run main inside a thread in main.py.


if __name__ == '__main__':
    bot = TelegramBot()
    bot.chat_id = 1706759043  # Matt's chat id.
    bot.main()

