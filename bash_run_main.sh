#!/usr/bin/env bash

cd ~/pyprojects/ironacer/
source /opt/homebrew/Caskroom/miniforge/base/bin/activate
conda activate pytorch_env
python main.py
conda deactivate
