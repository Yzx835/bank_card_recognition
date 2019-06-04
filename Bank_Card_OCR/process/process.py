import os
import cv2 as cv
import numpy as np
from data.constants import *


def move(image, name, _dir):
    h, w = image.shape[0:2]
    cv.imwrite(_dir + name[0] + "m1" + name, image[2:h, 0:w])
    cv.imwrite(_dir + name[0] + "m2" + name, image[0:h - 2, 0:w])
    return image[2:h, 0:w], image[0:h - 2, 0:w]


def project(image, name, _dir):
    h, w = image.shape[0:2]
    src_points = np.array([[0, 0], [0, h], [w, 0], [w, h]], np.float32)
    for i in range(0, 6, 2):
        for j in range(0, 6, 2):
            if (i == 0 and j == 0):
                continue
            dst_points = np.array([[0, 0], [0, h], [w, -i], [w, h + j]], np.float32)
            mat = cv.getPerspectiveTransform(src_points, dst_points)
            resimage = cv.warpPerspective(image, mat, (w, h))
            cv.imwrite(_dir + name[0] + "p1" + str(i) + str(j) + name, resimage)
            dst_points = np.array([[-i, 0], [0, h], [w + j, 0], [w, h]], np.float32)
            mat = cv.getPerspectiveTransform(src_points, dst_points)
            resimage = cv.warpPerspective(image, mat, (w, h))
            cv.imwrite(_dir + name[0] + "p2" + str(i) + str(j) + name, resimage)
            dst_points = np.array([[0, 0], [-i, h], [w, 0], [w + j, h]], np.float32)
            mat = cv.getPerspectiveTransform(src_points, dst_points)
            resimage = cv.warpPerspective(image, mat, (w, h))
            cv.imwrite(_dir + name[0] + "p3" + str(i) + str(j) + name, resimage)
            dst_points = np.array([[0, -i], [0, h + j], [w, 0], [w, h]], np.float32)
            mat = cv.getPerspectiveTransform(src_points, dst_points)
            resimage = cv.warpPerspective(image, mat, (w, h))
            cv.imwrite(_dir + name[0] + "p4" + str(i) + str(j) + name, resimage)


def blur(image, name, _dir):
    resimage = cv.GaussianBlur(image, (5, 5), 0)
    cv.imwrite(_dir + name[0] + "/b" + name, resimage)
    return resimage


def cut(count, name, image, _dir):
    size = image.shape
    eph = size[0]
    epw = int(size[1] / 4)
    for i in range(0, 4):
        cv.imwrite(_dir + str(name[i]) + "_" + str(count[str(name[i])]) + ".png", image[0:eph, epw * i:epw * (i + 1)])
        count[str(name[i])] = count[str(name[i])] + 1


def cut_all(raw_dir, each_dir):
    elm_list = os.listdir(raw_dir)
    num = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "_": 0}
    for elm in elm_list:
        cut(num, elm, cv.imread(raw_dir + elm), each_dir)


def enlarge(each_dir, enlarge_dir):
    elm_list = os.listdir(each_dir)
    for elm in elm_list:
        src = cv.imread(each_dir + elm)
        cv.imwrite(enlarge_dir + elm, src)
        src1, src2 = move(blur(src, elm, enlarge_dir), "b" + elm, enlarge_dir)
        src3, src4 = move(src, elm, enlarge_dir)
        project(blur(src, elm, enlarge_dir), "b" + elm, enlarge_dir)
        project(src3, "m1" + elm, enlarge_dir)
        project(src4, "m2" + elm, enlarge_dir)
        project(src1, "m1b" + elm, enlarge_dir)
        project(src2, "m2b" + elm, enlarge_dir)


if __name__ == "__main__":
    cut_all(DATA_PROCESS_RAW_PATH, DATA_PROCESS_EACH_PATH)
    enlarge(DATA_PROCESS_EACH_PATH, DATA_PROCESS_ENLARGE_PATH)
