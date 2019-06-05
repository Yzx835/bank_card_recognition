import sys
import os
sys.path.append("../")
import cv2 as cv
import recognition.number_recognize as number_recognize
from data.constants import *
import locate.card_locate as card_locate
import locate.number_locate as number_locate


src_path = DATA_LOCATE_RESULT_PATH


def line_recognize(image):

    image = cv.resize(image, (0, 0), fx=48. / image.shape[0], fy=48. / image.shape[0])

    image_height, image_width = image.shape[0:2]
    number_height = image_height
    number_width = int(number_height * (32. / 48.))
    temp_width = 0
    number_scole = max(int(number_width * (1 / 16)), 1)
    space_scole = max(int(number_width * (1 / 16)), 1)
    __output = ""
    while temp_width + number_width <= image_width:
        sub_image = image[0:image_height, int(temp_width):int(temp_width + number_width)]
        # cv.imshow("image", sub_image)
        # cv.waitKey(0)
        number = number_recognize.number_recognize(sub_image)
        __output += number
        if number == "_":
            temp_width += space_scole
        else:
            temp_width += number_scole
        # print(number)
    __output = __output[0] + __output + __output[-1]
    _output = ""
    temp = 0
    while temp < len(__output):
        if __output[temp] == '_':
            temp += 1
            continue
        numbers = ""
        while temp < len(__output):
            if __output[temp] == '_':
                temp += 1
                break
            numbers += __output[temp]
            temp += 1
        if len(numbers) < 2:
            continue
        number_dir = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        for num in numbers:
            number_dir[num] += 1
        num_got = "_"
        num_count = 0
        for num in number_dir:
            if number_dir[num] > num_count:
                num_got = num
                num_count = number_dir[num]
        _output += num_got
    return _output


def card_recognize(image):
    hough_image = card_locate.hough_lines(image, None)
    line_image = number_locate.tophat_locate(hough_image, None)
    return line_recognize(line_image)


if __name__ == '__main__':
    file_list = os.listdir(src_path)
    for elm in file_list:
        image = cv.imread(src_path + elm)
        output = line_recognize(image)
        print(elm + " : " + output)
