# Apply morphology of image : opening, closing, eroding, dilating, ...

import cv2
import numpy as np
import copy

def condition_rect(rect) :
    x, y, w, h = rect

    threshold_width = 20
    threshold_height = 10
    threshold_S = 2000
    threshold_x = 80
    if w > 2 * h:
        return True

    return False

def dectect_letters(img):
    bound_rects = []
    #img_ = copy.deepcopy(img)
    #_, img_ = cv2.threshold(img_, 100, 255, 1)
    #cv2.imshow("hehe", img_)
    #cv2.waitKey(0)
    #img_ = cv2.cvtColor(img, 7)
    img_ = img[:, :, 0]
    #should try more detect function
    #img_ = cv2.medianBlur(img_, 5)
    img_gray = cv2.GaussianBlur(img_, (5, 5), 0) # more effective
    #img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    #tmp = cv2.split(img_)
    #img_gray = tmp[0]
    img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1)
    img_sobel = cv2.Sobel(img_sobel, cv2.CV_8U, 1, 0)
    cv2.imshow("sobel", img_sobel)
    #img_sobel = cv2.Laplacian(img_gray, -1, 3)
    _, img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_GRADIENT, element)
    #cv2.imshow("morphology", img_threshold)
    contours, _ = cv2.findContours(img_threshold, 0, 1)

    for contour in contours:
        if len(contour) > 100:
            approx = cv2.approxPolyDP(contour, 2 , True)
            app_rect = cv2.boundingRect(approx)
            print app_rect
            x, y, w, h = app_rect

            if condition_rect(app_rect) is True:
                bound_rects.append(app_rect)

    return bound_rects
