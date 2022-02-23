#!/usr/bin/env python3

import cv2
import numpy as np
import sys


PATH_TO_OPENFACE_LIB_DIR = "."
PATH_TO_OPENFACE_MODEL_AU_ROOT_DIR = ".."
PATH_TO_IMAGE = "image.jpg"


sys.path.append(PATH_TO_OPENFACE_LIB_DIR)
import openface


def getCameraIntrinsics(image_shape):
	cx = image_shape[1] / 2.0
	cy = image_shape[0] / 2.0
	f = cx / 640.0 * 500 + cy / 480.0 * 500
	return f, f, cx, cy


# once at program start
# prepare openface: set camera inteinsics and optional argument string (check code for more information)
op = openface.OpenFace(PATH_TO_OPENFACE_MODEL_AU_ROOT_DIR)


# in program loop
# load image and resize image
image = cv2.imread(PATH_TO_IMAGE)
image = cv2.resize(image, (int(2000*image.shape[1]/image.shape[0]), 2000))	# openface works well on images with height 2000pxl
print("image shape:", image.shape, image.dtype)

# detect landmarks, aus, etc
success = op.detect(image, getCameraIntrinsics(image.shape))

# access data
if success:
	print("pose: ", op.pose)
	print("AU size:", len(op.au), len(op.au_binary))
	print("landmark size:", len(op.landmark_data), len(op.landmark_visibility), op.confidence)
	landmarks_x = op.landmark_data[:68]
	landmarks_y = op.landmark_data[68:]
	for x, y in zip(landmarks_x, landmarks_y):
		cv2.circle(image, (int(x), int(y)), 4, (127,255,0), -1)
image = cv2.resize(image, (int(900*image.shape[1]/image.shape[0]), 900))	# scale output to height 900pxl
cv2.imshow("test", image)
cv2.waitKey()
