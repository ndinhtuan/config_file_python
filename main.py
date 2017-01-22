import json
import cv2
from processWithConfig import processWithConfig

with open('config.json', 'r') as f:
    config = json.load(f)

img = cv2.imread("mu.jpg")
dst = processWithConfig(img, config)
cv2.imshow("tuan", dst)
cv2.waitKey(0)
