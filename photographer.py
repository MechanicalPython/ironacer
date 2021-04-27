#! /usr/local/bin/python3.9


from telegram.ext import Updater, CommandHandler, Filters, MessageHandler
import camera

# todo - Docs, send photo to only the one who asked for it, increasing robustness and control,


class TelegramBot:
    def __init__(self):
        self.token = open('telegram_token', 'r').read()
        self.updater = Updater(self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.bot = self.updater.bot
        self.camera = camera.Camera()
        self.mb_chat_id = 1706759043
        self.jb_chat_id = 1738114081

    @staticmethod
    def start(update, context):
        update.message.reply_text('start command received. /photo to take photos')

    @staticmethod
    def ping(update, context):
        update.message.reply_text('Ping')

    @staticmethod
    def help(update, context):
        update.message.reply_text('/photo to take 60 photos. \n/ping to check connection.')

    def take_photo(self, update, context):
        update.message.reply_text('Taking images...')
        last_photo_path = self.camera.take_burst(60)
        update.message.reply_text('Finished taking 60 images.')
        self.bot.sendPhoto(self.mb_chat_id, open(last_photo_path, 'rb'), timeout=300)
        self.bot.sendPhoto(self.jb_chat_id, open(last_photo_path, 'rb'), timeout=300)

    def take_one_photo(self, update, context):
        update.message.reply_text('Taking test image...')
        last_photo_path = self.camera.take_photo()
        self.bot.sendPhoto(self.mb_chat_id, open(last_photo_path, 'rb'), timeout=300)
        self.bot.sendPhoto(self.jb_chat_id, open(last_photo_path, 'rb'), timeout=300)

    def error(self, update, context):
        update.message.reply_text(f'An error occurred: {context}')

    def main(self):
        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('photo', self.take_photo))
        self.dispatcher.add_handler(CommandHandler('help', self.help))
        self.dispatcher.add_handler(CommandHandler('ping', self.ping))
        self.dispatcher.add_handler(CommandHandler('test_image', self.take_one_photo))

        self.dispatcher.add_error_handler(self.error)

        # start your shiny new bot
        self.updater.start_polling()

        # run the bot until Ctrl-C
        self.updater.idle()


if __name__ == '__main__':
    TelegramBot().main()
