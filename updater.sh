#!/usr/bin/env bash

cd /home/pi/ironacer
git remote update && git status -uno | grep -q 'Your branch is behind' && changed=1
if [ $changed = 1 ]; then
    git pull
    reboot;
else
    echo "Up-to-date"
fi