import cv2 as cv
import numpy as np
import locate.locate_funtion as loc


def mysobel(image):
    sx = cv.Sobel(image, cv.CV_16S, 1, 0)
    sy = cv.Sobel(image, cv.CV_16S, 0, 1)
    absx = cv.convertScaleAbs(sx)
    absy = cv.convertScaleAbs(sy)
    return absx, absy


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
        return ((c1 * b2 - c2 * b1) / det), ((a1 * c2 - a2 * c1) / det)


def hough_lines(image, name, card_dir):
    image = loc.resize_as_card(image)
    image_copy = loc.resize_as_card(image)
    image = loc.linear_gray(image)
    image = cv.GaussianBlur(image, (5, 5), 0)
    pox, poy = mysobel(image)

    pox = cv.threshold(pox, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    pox = cv.dilate(pox, None)
    hogx = cv.HoughLines(pox, 1, np.pi / 180, 150)
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
    hogy = cv.HoughLines(poy, 1, np.pi / 180, 300)
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
        if (abs(xminh[0][1] - np.pi / 2) < abs(hog_one[0][1] - np.pi / 2)):
            xminh = hog_one

    xmaxh = xmax[0]
    for hog_one in xmax:
        if (abs(xmaxh[0][1] - np.pi / 2) < abs(hog_one[0][1] - np.pi / 2)):
            xmaxh = hog_one

    yminh = ymin[0]
    for hog_one in ymin:
        if (abs(yminh[0][1] - np.pi / 2) > abs(hog_one[0][1] - np.pi / 2)):
            yminh = hog_one

    ymaxh = ymax[0]
    for hog_one in ymax:
        if (abs(ymaxh[0][1] - np.pi / 2) > abs(hog_one[0][1] - np.pi / 2)):
            ymaxh = hog_one

    x1, y1 = get_intsec(xminh, yminh)
    x2, y2 = get_intsec(xmaxh, yminh)
    x3, y3 = get_intsec(xminh, ymaxh)
    x4, y4 = get_intsec(xmaxh, ymaxh)

    src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.float32)
    dst_points = np.array([[0, 0], [loc.CARD_WIDTH, 0], [0, loc.CARD_HEIGHT], [loc.CARD_WIDTH, loc.CARD_HEIGHT]],
                          np.float32)

    mat = cv.getPerspectiveTransform(src_points, dst_points)
    image_copy = cv.warpPerspective(image_copy, mat, (loc.CARD_WIDTH, loc.CARD_HEIGHT))
    cv.imwrite(card_dir + name, image_copy)
