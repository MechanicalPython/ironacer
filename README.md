# Ironacer
A project to deter squirrels from eating the garden acer by using darknet's yolo object detection and a bluetooth
controlled water hose. 

## General structure
Take a photo every second, analyse the photo to squirrels, darknet_detect the location of the squirrel, 
operate the anti-squirrel measure: darknet_detect, find, strike. 


## Finding the squirrel. 
From a single image, to find the location of the squirrel I need the distance from the camera, and the angle from 
the center the squirrel is. 

### Distance
The height of a squirrel is relatively consistent. So, in theory, the short side of the bounding box should be the 
consistent height of the squirrel. The 

### Angle
pi camera: 62.2 degrees horizontal, 48.8 degrees vertical.