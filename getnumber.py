import cv2 as cv
import numpy as np
import copy
from matplotlib import pyplot as plt

# 银行卡的高度和宽度（单位：像素）
CARD_WIDTH = 674
CARD_HEIGHT = 425
# 两个闭运算算法所用的核，一个用于x轴，一个用于y轴。
rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (36, 12))
rect_kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (20, 10))

# function:
# 对图片进行缩放，贴近卡片
# 通过等比例缩放，将原图变成宽为CARD_WIDTH或者高为CARD_HEIGHT的图片（取更大的那张）
# parameters:
# image - 需要处理的原图
# return:
# resimage - 经过缩放后的图
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

# function:
# 对图片进行灰度化和线性拉伸
# parameters:
# image - 需要处理的原图
# return:
# lineal - 经过灰度化和线性拉伸后的图
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

# function:
# 顶帽算法，核使用rect_kernel
# parameters:
# image - 需要处理的原图
# return:
# morphimage - 顶帽算法处理后的图
def morph(image):
    return cv.morphologyEx(image, cv.MORPH_TOPHAT, rect_kernel)

# function:
# sobel算法提取x轴边缘，并进行核为rect_kernel的闭运算，
# 最后进行二值化，得到黑白图
# parameters:
# image - 需要处理的原图
# return:
# thresh - 顶帽算法处理后的图
def sobelx(image):
    sx = cv.Sobel(image, cv.CV_16S, 1, 0)
    absx = cv.convertScaleAbs(sx)
    absx = cv.morphologyEx(absx, cv.MORPH_CLOSE, rect_kernel)
    thresh = cv.threshold(absx, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    return thresh

# function:
# sobel算法提取y轴边缘，并进行核为rect_kernel2的闭运算，
# 最后进行二值化，得到黑白图
# parameters:
# image - 需要处理的原图
# return:
# thresh - 顶帽算法处理后的图
def sobely(image):
    sy = cv.Sobel(image, cv.CV_16S, 1, 0)
    absy = cv.convertScaleAbs(sy)
    absy = cv.morphologyEx(absy, cv.MORPH_CLOSE, rect_kernel2)
    thresh = cv.threshold(absy, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    return thresh

# function:
# 计算x轴方向的投影，并将水平投影中黑色像素的个数存到mapar数组中返回。
# 被注释掉的内容可以用于绘制直观的水平投影图，但是会修改image原图，慎用。
# parameters:
# image - 需要处理的原图
# return:
# mapar - 列表，index为x坐标，value为此坐标下纵列的黑色像素个数
def projectx(image):
    height, width = image.shape[0:2]
    mapar = []
    for i in range(height):
        tmp = 0
        for j in range(width):
            if (image[i, j] == 0):
                tmp = tmp + 1
                #image[i, j] = 255
        mapar.append(tmp)

    #for i in range(height):
    #    for j in range(mapar[i]):
    #       image[i, j] = 0

    return mapar

# function:
# 计算y轴方向的投影，并将水平投影中黑色像素的个数存到mapar数组中返回。
# 被注释掉的内容可以用于绘制直观的水平投影图，但是会修改image原图，慎用。
# parameters:
# image - 需要处理的原图
# return:
# mapar - 列表，index为y坐标，value为此坐标下横行的黑色像素个数
def projecty(image):
    height, width = image.shape[0:2]
    mapar = []
    for j in range(width):
        tmp = 0
        for i in range(height):
            if (image[i, j] == 0):
                tmp = tmp + 1
                #image[i, j] = 255
        mapar.append(tmp)

    #for j in range(width):
    #    for i in range(mapar[j]):
    #        image[i, j] = 0

    return mapar

# function:
# 在hough变换下的众多直线中，选择出我们所需要的两条横向直线，
# 并且将其画到image原图上。
# parameters:
# image - 需要处理的原图
# whlist - 垂直投影的黑色像素个数数组
# return:
# y1, y2 - 两条横向直线的纵坐标，y1 < y2.
def draw_linex(image, whlist):
    uplist = []
    downlist = []
    for i in range(len(whlist) - 10):
        if (whlist[i + 10] - whlist[i] > 100):
            # cv.line(gray, (0, i), (500, i), 255, 2)
            uplist.append(i)
        elif (whlist[i] - whlist[i + 10] > 100):
            # cv.line(gray, (0, i), (500, i), 0, 2)
            downlist.append(i)

    midline = (image.shape[0] / 2)
    tmpvalue = 800
    tmpi = 0

    for i in range(len(downlist)):
        if (abs(downlist[i] - midline) <= tmpvalue):
            tmpvalue = abs(downlist[i] - midline)
            tmpi = i

    for i in range(len(uplist) - 1):
        if (uplist[i] - downlist[tmpi] > 20 and uplist[i + 1] - uplist[i] > 10):
            tmpvalue = (int)(uplist[i])
            #print(downlist[tmpi] - uplist[i])
            if (uplist[i] - downlist[tmpi] < 40):
                downlist[tmpi] = uplist[i] - 40
            elif (uplist[i] - downlist[tmpi] > 50):
                downlist[tmpi] = uplist[i] - 50
            break

    cv.line(image, (0, (int)(downlist[tmpi]) + 5), (image.shape[1], (int)(downlist[tmpi]) + 5), 0, 2)
    cv.line(image, (0, (int)(tmpvalue) + 5), (image.shape[1], (int)(tmpvalue) + 5), 255, 2)
    return ((int)(downlist[tmpi]) + 5), ((int)(tmpvalue) + 5)

# function:
# 在hough变换下的众多直线中，选择出我们所需要的两条纵向直线，
# 并且将其画到image原图上。该函数本来想用于提取卡号的纵向边界，
# 但是并没有完成，现在没有什么有用的功能。
# parameters:
# image - 需要处理的原图
# whlist - 水平投影的黑色像素个数数组
# return:
# None
def draw_liney(image, whlist):
    uplist = []
    downlist = []
    for i in range(len(whlist) - 10):
        if (whlist[i + 10] - whlist[i] > 100):
            cv.line(image, (i, 0), (i, 500), 255, 2)
            uplist.append(i)
        elif (whlist[i] - whlist[i + 10] > 100):
            cv.line(image, (i, 0), (i, 500), 0, 2)
            downlist.append(i)
    '''
    midline = (image.shape[0] / 2)
    tmpvalue = 800
    tmpi = 0

    for i in range(len(downlist)):
        if (abs(downlist[i] - midline) <= tmpvalue):
            tmpvalue = abs(downlist[i] - midline)
            tmpi = i

    cv.line(image, (0, downlist[tmpi] + 5), (image.shape[1], downlist[tmpi] + 5), 0, 2)

    for i in range(len(uplist) - 1):
        if (uplist[i] - downlist[tmpi] > 20 and uplist[i + 1] - uplist[i] > 10):
            tmpvalue = i
            # print(downlist[tmpi] - uplist[i])
            break

    cv.line(image, (0, uplist[tmpvalue] + 5), (image.shape[1], uplist[tmpvalue] + 5), 255, 2)
'''

