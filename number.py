import cv2 as cv
import numpy as np
import getnumber
import os

list = os.listdir("test_images2/")
for elm in list:
    src = cv.imread("test_images2/" + elm)
    src = getnumber.resize_as_card(src)
    gray = getnumber.linear_gray(src)
    tophat = getnumber.morph(gray)
    sobimx = getnumber.sobelx(tophat)
    needlist = getnumber.projectx(sobimx)
    aup, adown = getnumber.draw_linex(gray, needlist)

    tophat = tophat[aup:adown, 0::]

    sobimy = getnumber.sobely(tophat)
    needlist = getnumber.projecty(sobimy)
    getnumber.draw_liney(gray, needlist)

    #cv.imshow("gray", gray)
    cv.imwrite("test_numbers/" + elm, src[aup:adown, 0::])
    #cv.waitKey(0)
    #cv.destroyAllWindows()