
import html
import json
import os
import traceback
import cv2

from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext

from ironacer import ROOT, DETECTION_REGION
from ironacer import utils

# todo - command to fire the hose and record the time before and after that.


class TelegramBot:
    """
    Standard telegram bot

    # Bot Commands
    help
    view - takes a photo of the current view with detection region box.
    saved - get number of saved images in detected/image

    # Methods
    send_message - takes text.
    send_photo - takes photo as bytes.
    send_video - takes video as bytes.
    send_doc - takes file_path.
    Max file size is 50MB or 10MB photo.

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

    @staticmethod
    def help(update, context):
        update.message.reply_text('/view to take a photo.\n'
                                  '/saved to get number of saved photos')

    def latest_view(self, update, context):
        """Accepts self.latest_frame which is a cv2 np.array()"""
        frame = utils.add_label_to_frame(self.latest_frame, [DETECTION_REGION])
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        update.message.reply_photo(frame, timeout=300)

    def get_current_number_of_images(self, update, context):
        update.message.reply_text(f"{len(os.listdir(f'{ROOT}/detected/image/'))} images currently saved")

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
        self.dispatcher.add_handler(CommandHandler('view', self.latest_view))
        self.dispatcher.add_handler(CommandHandler('saved', self.get_current_number_of_images))
        self.dispatcher.add_error_handler(self.error)

        self.updater.start_polling()
        # self.updater.idle()  # Comment out to run main inside a thread in main.py.


if __name__ == '__main__':
    bot = TelegramBot()
    bot.chat_id = 1706759043  # Matt's chat id.
    bot.main()

