import cv2 as cv
import locate.locate_funtion as loc


def tophat_locate(image, name):
    image = loc.resize_as_card(image)
    gray = loc.linear_gray(image)
    top_hat = loc.morph(gray)
    sobimx = loc.sobelx(top_hat)
    need_list = loc.projectx(sobimx)
    aup, adown = loc.draw_linex(gray, need_list)
    
    return image[aup:adown, 0::]
