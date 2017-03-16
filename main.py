import json
import cv2
from processWithConfig import processWithConfig
import os
import numpy as np
import copy
from perspective import four_point_transform
from extract_information import dectect_letters

def getFilesNameInDir(name_dir) :

    os.chdir(name_dir)
    allFiles = [os.path.abspath(d) for d in os.listdir('.') if os.path.isfile(os.path.join(name_dir, d))]

    return allFiles
#load file config json

def pre_processing (img) :

    dst = img;
    img_yuv = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
    cv2.imshow("yuv", dst)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    cv2.imshow("equal", img_yuv)
    dst = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow("BGR", dst)
    dst = cv2.cvtColor(dst, 7)
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #dst = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, element, borderType=cv2.BORDER_DEFAULT)
    dst = cv2.GaussianBlur(dst, (5, 5), 0)
    #dst = cv2.medianBlur(dst, 5)
    #dst = cv2.Canny(dst, 75, 200)
    dst = cv2.Laplacian(dst, -1, 3)
    #ret, dst = cv2.threshold(dst, 20, 255, 3)
    cv2.imshow("edge", dst)
    return dst

def draw_contours(img_src, img_edges) :
    img_tmp = copy.deepcopy(img_edges)
    contours, hi = cv2.findContours(img_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    screenCnt = None
    # loop over the contours
    for c in contours:
    	# approximate the contour
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    	# if our approximated contour has four points, then we
    	# can assume that we have found our screen
    	if len(approx) == 4:
    	    screenCnt = approx
    	    break

    cv2.drawContours(img_src, [screenCnt], -1, (0, 255, 0), 2)
    return screenCnt

def bound_information(img):

        bound_rects = dectect_letters(img)
        bounded_img = copy.deepcopy(img)

        for bound in bound_rects:
            x, y, w, h = bound
            cv2.rectangle(bounded_img, (x - 5, y - 5), (x+w, y+h),(0, 255, 0), 2)
        return bounded_img

with open('config.json', 'r') as f:
    config = json.load(f)
    print config

allFiles = getFilesNameInDir('/home/tuan/Desktop/IdentityCard/SourceImg')
i = 0
fail = 0

for name_img in allFiles :
    i += 1

    img = cv2.imread(name_img)
    img_src = copy.deepcopy(img)
    if img is None:
        print "Not open image."
        continue

    img_config = pre_processing(img);
    screenCnt = draw_contours(img_src, img_config)
    if screenCnt is None:
        fail+=1
        print "Not found contour"
        continue
    pts = screenCnt.reshape(4, 2)
    #warped_img = four_point_transform(img, pts)
    #bounded_img = bound_information(warped_img)

    #cv2.imshow("warped_img", warped_img)
    #cv2.imshow("bounded_img", bounded_img)
    #cv2.imshow("pre_processing", img_config)
    cv2.imshow("src", img_src)
    s = "/home/tuan/Desktop/IdentityCard/Result/{}.jpg".format(i)
    #cv2.imwrite(s, bounded_img)
    cv2.waitKey(0)

print "{} - {}".format(fail, i)
