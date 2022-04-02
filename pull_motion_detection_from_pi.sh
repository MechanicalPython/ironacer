#!/usr/bin/env bash

rsync --remove-source-files -avz "pi@ironacer.local:~/ironacer/*.zip" ~/
