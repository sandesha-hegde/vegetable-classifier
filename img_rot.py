import numpy as np
import cv2
import math


image = cv2.imread('/home/techvamp/Documents/Project/A/IMG_20180415_201500.jpg')
image = cv2.resize(image, (0, 0), fx=0.15, fy=0.15)

h, w = image.shape[:2]
image_center = (w / 2, h / 2)

rot_image = cv2.getRotationMatrix2D(image_center, 90, 1)
radians = math.radians(90)
sin = math.sin(radians)
cos = math.cos(radians)

bound_w = int((h * abs(sin)) + (w * abs(cos)))
bound_h = int((h * abs(cos)) + (w + abs(sin)))

rot_image[0, 2] += ((bound_w / 2) - image_center[0])
rot_image[1, 2] +=((bound_h / 2) - image_center[1])

rot_image = cv2.warpAffine(image, rot_image, (bound_w, bound_h))

cv2.imshow('hello', rot_image)
cv2.waitKey(0)

