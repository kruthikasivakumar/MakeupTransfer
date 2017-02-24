import cv2
import numpy as np
import stasm
import sys

def form_cnt( list_points, in_dict):
    out_cnt = np.zeros((len(list_points), 1, 2))
    for idx, point in enumerate(list_points):
        out_cnt[idx][0][0] = in_dict[point]['x']
        out_cnt[idx][0][1] = in_dict[point]['y']

    return out_cnt

in_name = sys.argv[1]

img = cv2.imread(in_name)
im_gray = cv2.imread(in_name, 0)

landmarks = stasm.search_single(im_gray)
landmark_dict = dict()
for idx, landmark in enumerate(landmarks):
    landmark_dict[str(idx)] = dict()
    landmark_dict[str(idx)]['x'] = landmark[0]
    landmark_dict[str(idx)]['y'] = landmark[1]

test_cnts = dict()
test_cnts['face'] = np.zeros((16, 1, 2))
test_cnts['left_eyebrow'] = np.zeros((6, 1, 2))
test_cnts['right_eyebrow'] = np.zeros((6, 1, 2))
test_cnts['left_eye'] = np.zeros((8, 1, 2))
test_cnts['right_eye'] = np.zeros((8, 1, 2))
test_cnts['upper_lip'] = np.zeros((10, 1,2))
test_cnts['lower_lip'] = np.zeros((10,1,2))
test_cnts['nose_tip'] = np.zeros((8, 1, 2))
test_cnts['lips'] = np.zeros((11, 1, 2))

#import pdb; pdb.set_trace()
face_points = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
test_cnts['face'] = form_cnt(face_points, landmark_dict).astype(np.int32)
left_eyebrow_points = ['16', '17', '18', '19', '20', '21']
test_cnts['left_eyebrow'] = form_cnt(left_eyebrow_points, landmark_dict).astype(np.int32)
right_eyebrow_points = ['24', '23', '22', '27', '26', '25']
test_cnts['right_eyebrow'] = form_cnt(right_eyebrow_points, landmark_dict).astype(np.int32)
left_eye_points = [str(i) for i in range(30,38)]
test_cnts['left_eye'] = form_cnt(left_eye_points, landmark_dict).astype(np.int32)
right_eye_points = [str(i) for i in range(40, 48)]
test_cnts['right_eye'] = form_cnt(right_eye_points, landmark_dict).astype(np.int32)
upper_lip_points = [str(i) for i in range(59, 69)]
test_cnts['upper_lip'] = form_cnt(upper_lip_points, landmark_dict).astype(np.int32)
lower_lip_points = ['59'] + [str(i) for i in range(69, 72)] + ['65'] + [str(i) for i in range(72, 77)]
test_cnts['lower_lip'] = form_cnt(lower_lip_points, landmark_dict).astype(np.int32)
nose_tip_points = [str(i) for i in range(51, 59)]
test_cnts['nose_tip'] = form_cnt(nose_tip_points, landmark_dict).astype(np.int32)
lip_points = [str(i) for i in range(59, 66)] + [str(i) for i in range(72, 77)]
test_cnts['lips'] = form_cnt(lip_points, landmark_dict).astype(np.int32)

mask = np.ones(im_gray.shape).astype(np.uint8)
cv2.drawContours(mask, [test_cnts['face']], 0 , 0 , -1)
mask = 1 - mask
mask = mask * 255
cv2.drawContours(mask, [test_cnts['left_eyebrow']], 0, 127, -1)
cv2.drawContours(mask, [test_cnts['right_eyebrow']], 0, 127, -1)
#cv2.drawContours(mask, [test_cnts['left_eye']], 0, 0, -1)
#cv2.drawContours(mask, [test_cnts['right_eye']], 0, 0, -1)
#cv2.drawContours(mask, [test_cnts['upper_lip']], 0, 0, -1)
#cv2.drawContours(mask, [test_cnts['lower_lip']], 0, 0, -1)
cv2.drawContours(mask, [test_cnts['nose_tip']], 0, 0, -1)
cv2.drawContours(mask, [test_cnts['lips']], 0, 0, -1)

cv2.imwrite(in_name.split('.')[0] + '_face_mask.png', mask)

#cv2.imshow('test_mask', mask)
#cv2.waitKey()
#cv2.namedWindow('test_mask')
