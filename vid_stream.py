"""
Streams the camera video via cv2 to get higher resolution images.
"""

import cv2


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open video device")
# Set properties. Each returns === True on success (i.e. correct resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3289)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2464)
# Read picture. ret === True on success

ret, frame = cap.read()
cv2.imwrite('cvframe', frame)

# Close device
cap.release()

