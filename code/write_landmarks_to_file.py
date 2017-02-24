import sys
import os
import dlib
import glob
from skimage import io
import stasm
import cv2
from util import *
import numpy as np
#import cv2

img = io.imread(sys.argv[1])
m,n,_ = img.shape
# #declare_windows('test', n,m)
points = list()
landmarks = stasm.search_single(cv2.imread(sys.argv[1],0))
for idx, landmark in enumerate(landmarks):
    points.append((int(landmark[0]), int(landmark[1])))
    #cv2.putText(img, str(idx), (int(landmark[0]), int(landmark[1])), fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.5, color=(0,255,0))
#cv2.imshow('test', img)
#cv2.waitKey()
#clear_windows('test')


#if len(sys.argv) != 3:
#    print(
#        "Give the path to the trained shape predictor model as the first "
#        "argument and then the directory containing the facial images.\n"
#        "For example, if you are in the python_examples folder then "
#        "execute this program by running:\n"
#        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
#        "You can download a trained facial shape predictor from:\n"
#        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
#    exit()

#predictor_path = sys.argv[1]
#img = io.imread(sys.argv[2])

#print sys.argv[2]

#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(predictor_path)

#points = list()

#dets = detector(img,1)
#for k,d_var in enumerate(dets):
#    shape = predictor(img,d_var)
#    for _,point in enumerate(shape.parts()):
#        points.append((point.x, point.y))

# #print points

with open('landmarks_file', 'w') as f_landmarks:
    for idx, point in enumerate(points):
        f_landmarks.write(str(point[0]) + ',' + str(point[1]) + '\n')
        #print(str(point[0]) + ',' + str(point[1]))
