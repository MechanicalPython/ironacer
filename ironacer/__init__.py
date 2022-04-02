"""
Define variables that are used throughout the package.
"""

from pathlib import Path
import os

FILE = Path(__file__).resolve()
ROOT = Path(os.path.abspath(FILE.parents[0])).parent  # Absolute path to ./ironacer/

YOLO_WEIGHTS = f'{ROOT}/yolov5n6_best.pt'
IMGSZ = 1280  # Only every going to be square as yolo needs square inputs.
DETECTION_REGION = [0, 180, 1280, 350]
MOTION_THRESH = 1000
