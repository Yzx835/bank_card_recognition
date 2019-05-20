import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def local_threshold(image):
    blur = cv.GaussianBlur(image, (5, 5), 0)
    gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, 6)
    #ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
    binary = cv.erode(binary, None)
    binary = cv.dilate(binary,None)
    cv.namedWindow("binary2", cv.WINDOW_NORMAL)
    cv.imshow("binary2", binary)

'''

'''

src = cv.imread('1.jpeg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)

blur = cv.GaussianBlur(src, (5,5), 0)
gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
canny = cv.Canny(gray, 30, 90)

bin, con, hie = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
temp = np.ones(canny.shape,np.uint8)*255
cv.drawContours(temp, con, -1, (0,255,0), 3)

size = temp.shape
height = size[0]
width = size[1]
mapar = []
for i in range(height):
    tmp = 0;
    for j in range(width):
        if (temp[i, j] == 0):
            tmp = tmp + 1
            temp[i, j] = 255
    mapar.append(tmp)

for i in range(height):
    for j in range(mapar[i]):
        temp[i, j] = 0;

plt.imshow(temp, cmap=plt.gray())
plt.show()

cv.namedWindow("canny", cv.WINDOW_NORMAL)
cv.imshow("canny", canny)
local_threshold(src)
cv.imwrite("out.png", temp)
cv.waitKey(0)
cv.destroyAllWindows()