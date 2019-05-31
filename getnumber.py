import cv2 as cv
import numpy as np
import copy
from matplotlib import pyplot as plt

CARD_WIDTH = 674
CARD_HEIGHT = 425
rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (36, 12))

def resize_as_card(image):
    h, w = image.shape[0:2];
    if (h * CARD_WIDTH > w * CARD_HEIGHT):
        pro = CARD_WIDTH / w
        resimage = cv.resize(image, (0, 0), fx=pro, fy=pro)
        h, w = resimage.shape[0:2]
        #cut = (int)((h - CARD_HEIGHT) / 2)
        #return resimage[cut:h-cut, 0:w]
    else:
        pro = CARD_WIDTH / w
        resimage = cv.resize(image, (0, 0), fx=pro, fy=pro)
        h, w = resimage.shape[0:2]
        #cut = (int)((w - CARD_WIDTH) / 2)
        #return resimage[0:h, cut:w-cut]
    return resimage

def linear_gray(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    rows = gray.shape[0]
    cols = gray.shape[1]
    lineal = copy.deepcopy(gray)
    flat = gray.reshape((cols * rows,)).tolist()
    A = min(flat)
    B = max(flat)
    for i in range(rows):
        for j in range(cols):
            lineal[i][j] = 255 * (gray[i][j] - A) / (B - A)
    return lineal

def morph(image):
    return cv.morphologyEx(image, cv.MORPH_TOPHAT, rect_kernel)

def sobel(image):
    sx = cv.Sobel(image, cv.CV_16S, 1, 0)
    sy = cv.Sobel(image, cv.CV_16S, 0, 1)
    absx = cv.convertScaleAbs(sx)
    #absy = cv.convertScaleAbs(sy)
    #cv.imshow("absx", absx)
    #cv.imshow("absy", absy)
    #return absx
    absx = cv.morphologyEx(absx, cv.MORPH_CLOSE, rect_kernel)
    thresh = cv.threshold(absx, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    return thresh

def project(image):
    height, width = image.shape[0:2]
    mapar = []
    for i in range(height):
        tmp = 0;
        for j in range(width):
            if (image[i, j] == 0):
                tmp = tmp + 1
                image[i, j] = 255
        mapar.append(tmp)

    for i in range(height):
        for j in range(mapar[i]):
            image[i, j] = 0;

    return image, mapar



src = cv.imread("test_images/9.jpeg")
cutim = resize_as_card(src)
gray = linear_gray(cutim)
tophat = morph(gray)
sobim = sobel(tophat)
proim, whlist = project(sobim)

uplist = []
downlist = []
for i in range(len(whlist) - 10):
    if (whlist[i+10] - whlist[i] > 100):
        #cv.line(gray, (0, i), (500, i), 255, 2)
        uplist.append(i)
    elif (whlist[i] - whlist[i+10] > 100):
        #cv.line(gray, (0, i), (500, i), 0, 2)
        downlist.append(i)

midline = (cutim.shape[0] / 2)
tmpvalue = 800
tmpi = 0

for i in range(len(downlist)):
    if (abs(downlist[i] - midline) <= tmpvalue):
        tmpvalue = abs(downlist[i] - midline)
        tmpi = i

cv.line(gray, (0, downlist[tmpi]+5), (cutim.shape[1], downlist[tmpi]+5), 0, 2)

for i in range(len(uplist) - 1):
    if (uplist[i] - downlist[tmpi] > 20 and uplist[i+1] - uplist[i] > 10):
        tmpvalue = i
        #print(downlist[tmpi] - uplist[i])
        break


cv.line(gray, (0, uplist[tmpvalue]+5), (cutim.shape[1], uplist[tmpvalue]+5), 255, 2)

cv.imshow("gray", gray)
cv.imshow("proim", proim)
cv.waitKey(0)
cv.destroyAllWindows()

