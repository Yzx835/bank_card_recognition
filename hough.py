import cv2 as cv
import copy
import numpy as np


def nothing(x):
    pass


src = cv.imread("test_images/5.jpeg")


gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
rows = gray.shape[0]
cols = gray.shape[1]

lineal = copy.deepcopy(gray)
flat = gray.reshape((cols * rows,)).tolist()
A = min(flat)
B = max(flat)
for i in range(rows):
    for j in range(cols):
        lineal[i][j] = 255 * (gray[i][j] - A) / (B - A)

blur = cv.GaussianBlur(lineal, (5, 5), 0)
canny = cv.Canny(blur, 30, 60)
canny = cv.dilate(canny, None)

hog = cv.HoughLinesP(canny, 1, np.pi/180, 400, minLineLength=5, maxLineGap=100)

for hog_one in hog:
    for x1, y1, x2, y2 in hog_one:
        if (x1 != x2):
            slope = (float)(y2 - y1) / (float)(x2 - x1)
            slope = abs(slope)
            if (slope <= 0.5):
                cv.line(src, (x1, y1), (x2, y2), (0,0,255), 1)

cv.imshow("hough", src)
cv.imshow("canny", canny)
cv.waitKey(0)
cv.destroyAllWindows()