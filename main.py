import json
import cv2
from processWithConfig import processWithConfig
import os
import numpy as np
import copy
from perspective import four_point_transform
from extract_information import dectect_letters
import pytesseract
from PIL import Image

def getFilesNameInDir(name_dir) :

    os.chdir(name_dir)
    allFiles = [os.path.abspath(d) for d in os.listdir('.') if os.path.isfile(os.path.join(name_dir, d))]

    return allFiles
#load file config json

def pre_processing (img) :

    dst = copy.deepcopy(img);

    #dst = cv2.cvtColor(dst, 7)
    dst = dst[:, :, 1]
    #cv2.imshow("img", dst)
    dst = cv2.equalizeHist(dst)
    #cv2.imshow("hist", dst)
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #dst = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, element, borderType=cv2.BORDER_DEFAULT)
    dst = cv2.GaussianBlur(dst, (5, 5), 0)
    #dst = cv2.medianBlur(dst, 5)
    #dst = cv2.adaptiveThreshold(dst,255,1,1,11,2)
    #dst = cv2.Canny(dst, 75, 200)
    dst = cv2.Laplacian(dst, -1, 3)
    #ret, dst = cv2.threshold(dst, 20, 255, 3)
    #cv2.imshow("edge", dst)
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
        box_information = []
        bounded_img = copy.deepcopy(img)

        for bound in bound_rects:
            x, y, w, h = bound
            box_information.append(img[y-5:y+h, x-5:x+w])
            cv2.rectangle(bounded_img, (x - 5, y - 5), (x+w, y+h),(0, 255, 0), 2)
        return bounded_img, box_information

def correction_string(str) :
    #print len(str)
    s = ''

    if len(str) >= 20 : return s
    for i in range(0, len(str)):
        if str[len(str) - 1 - i] >= '0' and str[len(str) - 1 - i] <= '9' :
            s += str[len(str) - 1 - i]
        else : s = ''

        if (len(s) == 9) : break

    s = s[::-1]
    return s;

with open('config.json', 'r') as f:
    config = json.load(f)
    print config

allFiles = getFilesNameInDir('/home/tuan/Desktop/IdentityCard/SourceImg')
i = 0
fail = 0

for name_img in allFiles :
    i += 1

    img = cv2.imread(name_img)

    #cv2.imshow("img_b", img[:, :, 0])
    #cv2.imshow("img_g", img[:, :, 1])
    #cv2.imshow("img_r", img[:, :, 2])

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
    warped_img = four_point_transform(img_src, pts)
    bounded_img, boxes = bound_information(warped_img)

    #cv2.imshow("warped_img", warped_img)
    cv2.imshow("bounded_img", bounded_img)
    #cv2.imshow("pre_processing", img_config)
    #cv2.imshow("src", img_src)

    cmt = ''

    for box in boxes:
        if box.shape[0] <= 0 or box.shape[1] <= 0:
            continue
        box = box[:, :, 0]
        #thres = cv2.threshold(box, 175, 255, 1)
        cv2.imshow("bound infor", box)
        #cv2.waitKey(1000)
        img = Image.fromarray(box)
        txt = pytesseract.image_to_string(img)
        print txt
        if txt == '':
            continue
        cmt = correction_string(txt)
        if(len(cmt) == 9) :
            break
    print "So CMT : {}".format(cmt)

    s = "/home/tuan/Desktop/IdentityCard/Result/{}.jpg".format(cmt)
    cv2.imwrite(s, warped_img)
    cv2.waitKey(0)

print "{} - {}".format(fail, i)
