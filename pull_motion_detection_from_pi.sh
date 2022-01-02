#!/usr/bin/env bash

rsync --remove-source-files -avz pi@ironacer.local:~/ironacer/motion_detected/ ~/motion_detected/
