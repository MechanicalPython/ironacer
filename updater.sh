#!/usr/bin/env bash

cd /home/pi/ironacer
if [[ `git status --porcelain` ]]; then
  git pull origin main
  reboot
else
  pass
fi
