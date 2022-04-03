#!/usr/bin/env bash

cd /home/pi/ironacer
git pull origin main
sudo systemctl restart ironacer
