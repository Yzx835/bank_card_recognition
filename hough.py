import cv2 as cv
import os
import numpy as np
import getnumber

def nothing(arg):
    pass

def mysobel(image):
    sx = cv.Sobel(image, cv.CV_16S, 1, 0)
    sy = cv.Sobel(image, cv.CV_16S, 0, 1)
    absx = cv.convertScaleAbs(sx)
    absy = cv.convertScaleAbs(sy)
    return absx, absy

def draw_line(image, line):
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image, (x1, y1), (x2, y2), 255, 2)

def get_intsec(line1, line2):
    for rho, theta in line1:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = x0 + 1000 * (-b)
        y1 = y0 + 1000 * (a)
        x2 = x0 - 1000 * (-b)
        y2 = y0 - 1000 * (a)
    for rho, theta in line2:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x3 = x0 + 1000 * (-b)
        y3 = y0 + 1000 * (a)
        x4 = x0 - 1000 * (-b)
        y4 = y0 - 1000 * (a)
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = x1 * y2 - x2 * y1
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = x3 * y4 - x4 * y3
    det = a1 * b2 - a2 * b1
    if (det == 0):
        return -1, -1
    else:
        return ((c1*b2-c2*b1)/det), ((a1*c2-a2*c1)/det)

list = os.listdir("test_images/")
for elm in list:
    src = cv.imread("test_images/" + elm)
    src = getnumber.resize_as_card(src)
    sss = getnumber.resize_as_card(src)
    src = getnumber.linear_gray(src)
    src = cv.GaussianBlur(src, (5, 5), 0)
    pox, poy = mysobel(src)

    pox = cv.threshold(pox, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    pox = cv.dilate(pox, None)
    hogx = cv.HoughLines(pox, 1, np.pi/180, 150)
    xmin = []
    xmax = []

    rholist = []
    for hog_one in hogx:
        for rho, theta in hog_one:
            if (theta < 0.1 or theta > np.pi - 0.1) and (abs(rho) < pox.shape[1] - 10):
                rholist.append(abs(rho))

    for hog_one in hogx:
        for rho, theta in hog_one:
            if ((theta < 0.1 or theta > np.pi - 0.1) and (abs(rho) < pox.shape[1] - 10)):
                if (abs(rho) < min(rholist) + 20):
                    xmin.append(hog_one)
                elif (abs(rho) > max(rholist) - 20):
                    xmax.append(hog_one)

    poy = cv.threshold(poy, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    poy = cv.dilate(poy, None)
    hogy = cv.HoughLines(poy, 1, np.pi/180, 300)
    rholist.clear()
    ymin = []
    ymax = []
    for hog_one in hogy:
        for rho, theta in hog_one:
            if (abs(theta - np.pi / 2) < 0.1) and (abs(rho) < pox.shape[0] - 10):
                rholist.append(abs(rho))

    for hog_one in hogy:
        for rho, theta in hog_one:
            if (abs(theta - np.pi / 2) < 0.1 and (abs(rho) < pox.shape[0] - 10)):
                if (abs(rho) < min(rholist) + 20):
                    ymin.append(hog_one)
                elif (abs(rho) > max(rholist) - 20):
                    ymax.append(hog_one)

    xminh = xmin[0]
    for hog_one in xmin:
        if (abs(xminh[0][1] - np.pi/2) < abs(hog_one[0][1] - np.pi/2)):
            xminh = hog_one

    xmaxh = xmax[0]
    for hog_one in xmax:
        if (abs(xmaxh[0][1] - np.pi/2) < abs(hog_one[0][1] - np.pi/2)):
            xmaxh = hog_one

    yminh = ymin[0]
    for hog_one in ymin:
        if (abs(yminh[0][1] - np.pi/2) > abs(hog_one[0][1] - np.pi/2)):
            yminh = hog_one

    ymaxh = ymax[0]
    for hog_one in ymax:
        if (abs(ymaxh[0][1] - np.pi / 2) > abs(hog_one[0][1] - np.pi / 2)):
            ymaxh = hog_one

    draw_line(src, xminh)
    draw_line(src, xmaxh)
    draw_line(src, yminh)
    draw_line(src, ymaxh)

    x1, y1 = get_intsec(xminh, yminh)
    x2, y2 = get_intsec(xmaxh, yminh)
    x3, y3 = get_intsec(xminh, ymaxh)
    x4, y4 = get_intsec(xmaxh, ymaxh)

    src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.float32)
    dst_points = np.array([[0, 0], [getnumber.CARD_WIDTH, 0], [0, getnumber.CARD_HEIGHT], [getnumber.CARD_WIDTH, getnumber.CARD_HEIGHT]], np.float32)

    mat = cv.getPerspectiveTransform(src_points, dst_points)
    sss = cv.warpPerspective(sss, mat, (getnumber.CARD_WIDTH, getnumber.CARD_HEIGHT))
    cv.imwrite("test_images2/" + elm, sss)
    '''
    cv.imshow("cnm", src)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''




'''
cv.namedWindow("res")
cv.createTrackbar('min','res', 0, 1000, nothing)
cv.createTrackbar('max','res', 0, 1000, nothing)
while(1):
    if cv.waitKey(1)&0xFF==27:
        break
    maxVal = cv.getTrackbarPos('max', 'res')
    minVal = cv.getTrackbarPos('min', 'res')
    canny = cv.Canny(poy, minVal, maxVal)
    canny = cv.dilate(canny, None)
    cv.imshow('res', canny)
'''