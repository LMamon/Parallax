import cv2
import numpy as np

left = cv2.imread("left.png")
right = cv2.imread("right.png")

left_map_x = np.load("config/camera/calibration/results/rectification/left_map_x.npy")
left_map_y = np.load("config/camera/calibration/results/rectification/left_map_y.npy")

right_map_x = np.load("config/camera/calibration/results/rectification/right_map_x.npy")
right_map_y = np.load("config/camera/calibration/results/rectification/right_map_y.npy")

left_rect = cv2.remap(left,
                    left_map_x,
                    left_map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)

right_rect = cv2.remap(right,
                    right_map_x,
                    right_map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)

cv2.imwrite("opencv_rectified_left.png", left_rect)
cv2.imwrite("opencv_rectified_right.png", right_rect)