#! /usr/local/bin/python3.7
"""
Telegram bot either had to be run in a parallel thread to be used a a real time 2-way interaction, or it can be used
to passively send information at pre-determined times.
"""

import html
import json
import logging
import traceback

from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext


class TelegramBot:
    def __init__(self):
        self.token = open('telegram_token', 'r').read()
        self.updater = Updater(self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.bot = self.updater.bot
        self.chat_id = -547385621
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def start(update, context):
        update.message.reply_text('start command received.')

    @staticmethod
    def help(update, context):
        update.message.reply_text('/photo to take a test photo. \n'
                                  '/ping to check connection.')

    def status(self, update, context):
        """
        Replies with current status of the camera:
        - Number of images in motion_detected
        - Time of last seen squirrel
        - Photo of current view?
        - Current mode that it's in? Pi_mode, inference, etc?
        """
        update.message.reply_text('')

    def send_photo(self, photo_path):
        self.bot.sendPhoto(self.chat_id, open(photo_path, 'rb'), timeout=300)

    def send_video(self, vid_path):
        self.bot.sendVideo(self.chat_id, open(vid_path, 'rb'), timeout=300)

    def error(self, update: object, context: CallbackContext):
        self.logger.error(msg="Exception while handling an update:", exc_info=context.error)
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
        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('help', self.help))
        self.dispatcher.add_handler(CommandHandler('status', self.status))

        self.dispatcher.add_error_handler(self.error)

        # start your shiny new bot
        self.updater.start_polling()

        # run the bot until Ctrl-C
        self.updater.idle()


if __name__ == '__main__':
    bot = TelegramBot()
    bot.send_photo('./test.jpg')
