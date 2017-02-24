import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2

if False:
#if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    #exit()

predictor_path = sys.argv[1]
img = io.imread(sys.argv[2])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
points = list()

[m,n,d] = img.shape
min_x = m
max_x = 0
min_y = n
max_y = 0
dets = detector(img,1)
for k,d_var in enumerate(dets):
    shape = predictor(img,d_var)
    min_x = min([dets[k].left(), min_x])
    max_x = max([dets[k].right(), max_x])
    min_y = min([dets[k].top(), min_y])
    max_y = max([dets[k].bottom(), max_y])
    for _,point in enumerate(shape.parts()):
        points.append((point.x, point.y))
        min_x = min([point.x, min_x])
        max_x = max([point.x, max_x])
        min_y = min([point.y, min_y])
        max_y = max([point.y, max_y])
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
    print("first min_x: {}, max_x: {}, min_y: {}, max_y: {}".format(min_x, max_x, min_y, max_y))
    print("points: ", points)

print("2nd min_x: {}, max_x: {}, min_y: {}, max_y: {}".format(min_x, max_x, min_y, max_y))
test_in = img[min_y:max_y+1, min_x:max_x+1,:]
dets_2 = detector(test_in, 1)
points_2 = list()
for idx, each_det in enumerate(dets_2):
    shape_2 = predictor(test_in, each_det)
    for _, point_2 in enumerate(shape_2.parts()):
        points_2.append((point_2.x, point_2.y))

(test_m, test_n, _) = test_in.shape
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", test_n, test_m)
#for idx, each_point in enumerate(points_2):
#    cv2.putText(test_in, str(idx), each_point, fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
#        fontScale = 0.4,
#        color = (255, 0,0))

cv2.imshow("output", test_in)
cv2.waitKey()
io.imsave(sys.argv[3] + ".png", test_in)