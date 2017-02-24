import numpy as np
import cv2
import dlib
from skimage import io
import sys
import os
import glob
from util import *

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
img_gray = cv2.imread(sys.argv[1], 0)

k = img.shape
height = k[0]
width = k[1]
output_fd = get_flandmarks(img, predictor_path, store = True)
#to_cnts = img[0:(output_fd['max_y'] + 20),:,:]
output_cnts = get_contours(img)
cnts = output_cnts['cnts']

min_x = min(output_fd['min_x'], output_cnts['min_x'])
#max_x = max(output_fd['max_x'], output_cnts['max_x'])
max_x = output_fd['max_x']
min_y = min(output_fd['min_y'], output_cnts['min_y'])
max_y = max(output_fd['max_y'], output_cnts['max_y'])

rect = (int(min_x), int(min_y), int(max_x), int(max_y))
mask = np.zeros(img.shape[:2], np.uint8)

bgd_m = np.zeros((1,65), np.float64)
fgd_m = np.zeros((1,65), np.float64)

cv2.grabCut(img, mask, rect, bgd_m, fgd_m, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 1) + (mask == 3), 255, 0)
map_binary = mask2.astype('float')

#io.imshow(map_binary)
#io.show()

inv_map_binary = (map_binary + 1)%2
dict_points = dict()
dict_points_x = dict()

[height, width] = img_gray.shape
cnts_map = np.zeros(img_gray.shape).astype('float')
for each_cnt in cnts:
    index = each_cnt.shape[0]
    for point in range(index):
        y = each_cnt[point][0][1]
        x = each_cnt[point][0][0]

        if x not in dict_points_x.keys():
            dict_points_x[x] = list()
        dict_points_x[x].append(y)

        if y not in dict_points.keys():
            dict_points[y] = list()
        dict_points[y].append(x)

        #import pdb; pdb.set_trace()
        #cnts_map[y-10:height-1, x] = 1.0

cnt_dict = dict()
cnt_dict = dict_points

for each_key in dict_points_x.keys():
    if len(dict_points_x[each_key]) > 1:
        y_list = dict_points_x[each_key]
        y_min = min(y_list)
        y_max = max(y_list)
        y_diff = y_max - y_min
        if(np.sum(cnts_map[y_min:y_max, each_key]) < y_diff):
            cnts_map[y_min:y_max, each_key] = 1.0

for each_key in cnt_dict.keys():
    if len(cnt_dict[each_key]) > 1:
        x_list = cnt_dict[each_key]
        x_min = min(x_list)
        x_max = max(x_list)
        x_diff = x_max - x_min
        if(np.sum(cnts_map[each_key, x_min:x_max]) < x_diff):
            cnts_map[each_key, x_min:x_max] = 1.0

temp_cnts_map = cnts_map
test_out = np.multiply(temp_cnts_map, inv_map_binary)
test_out = 127*test_out

#import pdb; pdb.set_trace()

test_out_temp = np.add(test_out, map_binary)
#io.imshow(test_out_temp)
#io.show()
test_fin = np.zeros((height, width, 3))
test_fin[:,:,0] = test_out_temp
test_fin[:,:,1] = test_out_temp
test_fin[:,:,2] = test_out_temp
io.imsave(sys.argv[1].split('.')[0] + '_trimap.png', test_fin.astype('uint8'))
