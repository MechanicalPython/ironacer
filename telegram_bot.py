#! /usr/local/bin/python3.7
"""
Telegram bot either had to be run in a parallel thread to be used a real time 2-way interaction, or it can be used
to passively send information at pre-determined times.
"""

import html
import json
import traceback
import os

from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext


class TelegramBot:
    def __init__(self):
        self.token = open(f'{os.path.dirname(__file__)}/telegram_token', 'r').read()
        self.updater = Updater(self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.bot = self.updater.bot
        self.chat_id = -547385621

    @staticmethod
    def start(update, context):
        update.message.reply_text('start command received.')

    @staticmethod
    def help(update, context):
        update.message.reply_text('/photo to take a test photo. \n'
                                  '/ping to check connection.')

    def send_message(self, text):
        self.bot.send_message(self.chat_id, text)

    def send_photo(self, photo_bytes):
        self.bot.sendPhoto(self.chat_id, photo_bytes, timeout=300)

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
        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('help', self.help))

        self.dispatcher.add_error_handler(self.error)

        # start your shiny new bot
        self.updater.start_polling()

        # run the bot until Ctrl-C
        self.updater.idle()


if __name__ == '__main__':
    bot = TelegramBot()
    bot.chat_id = 1706759043  # Matt's chat id.
    bot.send_doc('./detected2022-02-05.zip')
    # bot.main()

