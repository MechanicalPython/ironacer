"""
Define variables that are used throughout the package.
"""

from pathlib import Path
import os
import configparser

FILE = Path(__file__).resolve()
ROOT = Path(os.path.abspath(FILE.parents[0])).parent  # Absolute path to ./ironacer

parser = configparser.ConfigParser()
parser.read(f'{ROOT}/settings.cfg')

YOLO_WEIGHTS = f"{ROOT}/{parser.get('Settings', 'YOLO_WEIGHTS')}"
IMGSZ = parser.getint('Settings', 'IMGSZ')    # Only every going to be square as yolo needs square inputs.
DETECTION_REGION = [int(i) for i in parser.get('Settings', 'DETECTION_REGION').split(',')]
MOTION_THRESH = parser.getint('Settings', 'MOTION_THRESH')
