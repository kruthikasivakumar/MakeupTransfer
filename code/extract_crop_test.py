import numpy as np
#import cv2
import dlib
from skimage import io
import sys
import os
import glob
from util import *

if __name__ == '__main__':
    if len(sys.argv) > 5:
        print(
            "Give the path to the trained shape predictor model as the first "
            "argument and then the directory containing the facial images.\n"
            "For example, if you are in the python_examples folder then "
            "execute this program by running:\n"
            "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
            "You can download a trained facial shape predictor from:\n"
            "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        exit()

    win_in = 'input'
    win_out = 'output'
    cnts_win = 'cnt_map'
    cropped_win = 'cropped_image'

    predictor_path = 'shape_predictor.dat'
    img = io.imread(sys.argv[1])

    k = img.shape
    height = k[0]
    width = k[1]
    output_fd = get_flandmarks(img, predictor_path)
    to_cnts = img[0:(output_fd['max_y'] + 20),:,:]
    output_cnts = get_contours(to_cnts)

    diff_land_min = float(output_fd['mid_x'] - output_fd['min_x'])
    diff_land_max = float(output_fd['max_x'] - output_fd['mid_x'])

#print diff_land_min
#print diff_land_max

#import pdb; pdb.set_trace()

    r_min = 0.75
    r_max = 0.70

    min_x = output_fd['mid_x'] - (diff_land_min/r_min)
    max_x = output_fd['mid_x'] + (diff_land_max/r_max)

    min_y = min(output_fd['min_y'], output_cnts['min_y']) - 10
    max_y = output_fd['max_y'] + 20


    min_x = int(clamp_val_min(min_x,0))
    max_x = int(clamp_val_max(max_x, width - 1))
    min_y = int(clamp_val_min(min_y, 0))
    max_y = int(clamp_val_max(max_y, height - 1))

    im_cropped = img[min_y:max_y, min_x:max_x, :]
    io.imsave(sys.argv[1].split('.')[0]+"_cr.png", im_cropped)
#show_image(im_cropped, cropped_win)




























# import pdb; pdb.set_trace()
# diff = float(-(output_cnts['min_x'] - output_fd['min_x']))/float(width)
# print (output_cnts['min_x'], output_fd['min_x'])
# if diff > 0 and diff > 1./6.:
#     min_x = output_fd['min_x'] - 30
# else:
#     min_x = min(output_cnts['min_x'], output_fd['min_x'])

# diff = float(output_cnts['max_x'] - output_fd['max_x'])/float(width)
# if diff > 0 and diff > 1./6.:
#     max_x = output_fd['max_x'] + 30
# else:
#     max_x = max(output_cnts['max_x'], output_fd['max_x'])
