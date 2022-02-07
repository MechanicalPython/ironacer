#!/usr/bin/env bash

cd /home/pi/ironacer
git remote update
if [[ `git status --porcelain` ]]; then
  git pull origin main
  reboot
else
  pass
fi
