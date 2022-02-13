# Ironacer

## Aim of the Ironacer
The mission statement of the project is to leverage object recognition to fire water at squirrels that enter the 
garden, in the service of non-lethally preventing them from eating the acer tree or digging up bulbs. 

## The parts
### Detection
The detection of the squirrels is handled by the Yolov5 object recongition algorithm found at: 
https://github.com/ultralytics/yolov5 and is powered by pytorch. 

The camera is run by a raspberry pi zero which provides a video stream. The inference of the images is then run on 
a mac mini in a conda environment to allow pytorch to run natively on apple silicon. 

### Deterrence
Use of a raspberry pi controlled solenoid valve to control the flow of water. 

### Information
Use of telegram to publish images of detected squirrels. 

## Project files
This project is a weird mix of https://github.com/ultralytics/yolov5 and my own code. 
camera.py - basically useless as it controlled the camera but that's done by stream.py
stream.py - runs on the pi and serves the camera video for find.py
find.py - class that reads the streaming video and runs motion detection on it of yolov5 inference. Saves images to 
motion_detected, results, and training_wheels. 
telegram_bot.py - runs the telegram bot to send images and videos of detected squirrels. 
utils.py - holds random one-off functions. 

## Installations
### Camera
To use cv2, you need to enable the legacy camera via raspi-config. 

### Conda env
Mostly following: https://towardsdatascience.com/yes-you-can-run-pytorch-natively-on-m1-macbooks-and-heres-how-35d2eaa07a83
```
brew install miniforge
conda init zsh
conda create --name pytorch_env python=3.8
conda activate pytorch_env
conda install pytorch torchvision torchaudio -c pytorch
```
Then run python detect.py etc inside that pytorch_env. 
I think that will work, this was written after I got it to all work and it wasn't straightforward. 


## Training runs and weights
All run and saved in yolov5/runs/train/exp{}/weights/{best.pt, last.pt}
exp - exp2 are trained with 640 image size. 
exp2 - exp5 are trained with 1280 image size. 

yolov5 has runs in gitignore. 


### Angle
pi camera: 62.2 degrees horizontal, 48.8 degrees vertical.


If it all goes wrong
removed models and utils from the root folder.



## Gadget mode for testing pi zero. 
32 bit install. 

internet sharing for Ethernet/Gadget on the mac. 

Now, edit the file called cmdline.txt. Look for rootwait, and add modules-load=dwc2,g_ether immediately after.

In config.txt, and append the following: dtoverlay=dwc2


sudo apt install python3-opencv and pip3 install opencv-python-headless
cv2 dependancies 
sudo apt-get install libcblas-dev
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev 
sudo apt-get install libqtgui4 
sudo apt-get install libqt4-test