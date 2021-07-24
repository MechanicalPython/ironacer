#! /usr/local/bin/python3.7

import html
import json
import logging
import traceback

from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext

import camera


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
        update.message.reply_text('start command received. /photo to take photos')

    @staticmethod
    def ping(update, context):
        update.message.reply_text('Ping')

    @staticmethod
    def help(update, context):
        update.message.reply_text('/photo to take a test photo. \n'
                                  '/ping to check connection.')

    def photo(self, update, context):
        update.message.reply_text('Taking test image...')
        cam = camera.Camera('photos')
        last_photo_path = cam.take_photo()
        cam.__exit__()
        self.bot.sendPhoto(self.chat_id, open(last_photo_path, 'rb'), timeout=300)

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
        self.dispatcher.add_handler(CommandHandler('ping', self.ping))
        self.dispatcher.add_handler(CommandHandler('test_image', self.photo))

        self.dispatcher.add_error_handler(self.error)

        # start your shiny new bot
        self.updater.start_polling()

        # run the bot until Ctrl-C
        self.updater.idle()


if __name__ == '__main__':
    TelegramBot().main()
