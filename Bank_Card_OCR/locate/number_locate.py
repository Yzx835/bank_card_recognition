import cv2 as cv
import locate.locate_funtion as loc


def tophat_locate(image, name, result_dir):
    image = loc.resize_as_card(image)
    gray = loc.linear_gray(image)
    top_hat = loc.morph(gray)
    sobimx = loc.sobelx(top_hat)
    need_list = loc.projectx(sobimx)
    aup, adown = loc.draw_linex(gray, need_list)

    cv.imwrite(result_dir + name, image[aup:adown, 0::])
