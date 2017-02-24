import cv2
import numpy as np
import dlib
from skimage import io
import sys
import os
import glob

def clear_windows(win_name):
    cv2.destroyAllWindows()
    cv2.namedWindow(win_name)

def declare_windows(win_name, win_width, win_height):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_width, win_height)

def get_contours(im):
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)
    edged = cv2.Canny(img_gray, 20, 100)
    cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    [m, n] = img_gray.shape
    min_x = m
    max_x = 0
    min_y = n
    max_y = 0
    for contour in cnts:
        length = contour.shape[0]
        for p_idx in range(length):
            y = contour[p_idx][0][1]
            x = contour[p_idx][0][0]
            min_y = min(y, min_y)
            max_y = max(y, max_y)
            min_x = min(x, min_x)
            max_x = max(x, max_x)
    
    output= dict()
    output['cnts'] = cnts
    output['min_x'] = min_x
    output['max_x'] = max_x
    output['min_y'] = min_y
    output['max_y'] = max_y

    return output

def get_flandmarks(im, p_path, store = False):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p_path)
    points = list()
    [m,n,d] = im.shape
    min_x = m
    max_x = 0
    min_y = n
    max_y = 0
    mid_x = 0
    mid_y = 0
    ctr = 0.
    dets = detector(im, 1)
    for k, d_var in enumerate(dets):
        shape = predictor(im, d_var)
        min_x = min([dets[k].left(), min_x])
        max_x = max([dets[k].right(), max_x])
        min_y = min([dets[k].top(), min_y])
        max_y = max([dets[k].bottom(), max_y])
        for pt_idx,point in enumerate(shape.parts()):
            if pt_idx == 27 or pt_idx == 28 or pt_idx == 29 or pt_idx == 30:
                mid_x = mid_x + point.x
                mid_y = mid_y + point.y
                ctr += 1
            points.append((point.x, point.y))
            min_x = min([point.x, min_x])
            max_x = max([point.x, max_x])
            min_y = min([point.y, min_y])
            max_y = max([point.y, max_y])

    if store is True:
    	with open('landmarks_file', 'w') as f_landmarks:
    		for idx, point in enumerate(points):
        		f_landmarks.write(str(point[0]) + ',' + str(point[1]) + '\n')

    min_y = float(min_y)/2.
    mid_x = float(mid_x)/ctr
    mid_y = float(mid_y)/ctr
    out = dict()
    out['min_x'] = min_x
    out['max_x'] = max_x
    out['min_y'] = min_y
    out['max_y'] = max_y
    out['mid_x'] = mid_x
    out['mid_y'] = mid_y
    if store is not True:
    	out['points'] = points
    #print "mid_x", mid_x

    return out

def clamp_val_min(val, c_val):
    if val < 0:
        val = c_val

    return val

def clamp_val_max(val, c_val):
    if val > c_val:
        val = c_val

    return val

def draw_contours(img, contours):
    window = 'cnts_window'
    [m, n, d] = img.shape
    for idx in range(len(contours)):
        cv2.drawContours(img, contours, idx, (0,255,0),3)
    declare_windows(window, n, m)
    cv2.imshow(window, img)
    cv2.waitKey()
    clear_windows(window)

def show_image(img, str_win):
    k = img.shape
    m = k[0]
    n = k[1]
    declare_windows(str_win, n, m)
    cv2.imshow(str_win, img)
    cv2.waitKey()
    clear_windows(str_win)    
