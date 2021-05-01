#! /usr/local/bin/python3.7

"""
Module to watch for squirrels using yolo and report back to either the shooter module (to be built) or a telegram bot
"""

from telegram_bot import TelegramBot
from utils import Camera

import cv2
import numpy as np


class Watcher:
    """

    """

    def __init__(self, cfg, classes, weights):
        self.cfg = cfg
        self.weights = weights
        self.classes = classes
        self.net = cv2.dnn.readNet(weights, cfg)


if __name__ == '__main__':
    image = cv2.imread('/Users/Matt/squirrel_photos/2021')
