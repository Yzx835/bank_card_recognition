import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def local_threshold(name, image):
    blur = cv.GaussianBlur(image, (5, 5), 0)
    gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, 6)
    #ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
    binary = cv.erode(binary, None)
    binary = cv.dilate(binary,None)
    cv.imwrite("images_out/" + elm, binary)

def canny(name, image):
    blur = cv.GaussianBlur(image, (5, 5), 0)
    gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    canny = cv.Canny(gray, 30, 90)
    cv.imwrite("images_out2/" + elm, canny)

def cut(count, name, image):
    size = image.shape
    eph = size[0]
    epw = int(size[1] / 4)
    for i in range(0,4):
        cv.imwrite("images_each/" + str(name[i]) + "_" + str(count[str(name[i])]) + ".png", image[0:eph, epw * i:epw * (i + 1)])
        count[str(name[i])] = count[str(name[i])] + 1;

list = os.listdir("images/")
num = {"0": 0, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0, "7":0, "8":0, "9":0, "_":0}
for elm in list:
    cut(num, elm, cv.imread("images/" + elm))

list = os.listdir("images_each/")
for elm in list:
    src = cv.imread("images_each/" + elm)
    #local_threshold(elm, src)
    #canny(elm, src)


