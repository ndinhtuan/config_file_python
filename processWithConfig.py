import cv2

def processWithFunction(img, name_function, list_arguments):

    dst = img;

    if (name_function == 'canny'):
        dst = cv2.Canny(dst, list_arguments[0], list_arguments[1])

    if (name_function == 'sobel'):
        dst = cv2.Sobel(dst, list_arguments[0], list_arguments[1], list_arguments[2])

    if (name_function == 'median_blur'):
        dst = cv2.medianBlur(dst, list_arguments[0]);
        
    return dst

def processWithConfig(img, config):

    list_function = config.keys()
    dst = img

    for i in range(len(list_function)):
        dst = processWithFunction(dst, list_function[i], config[list_function[i]])

    return dst
