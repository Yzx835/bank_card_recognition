import os
import cv2 as cv
import numpy as np

def move(image, name):
    h, w = image.shape[0:2]
    cv.imwrite("images_enlarge/m1" + name, image[2:h, 0:w])
    cv.imwrite("images_enlarge/m2" + name, image[0:h-2, 0:w])
    return image[2:h, 0:w], image[0:h-2, 0:w]

def project(image, name):
    h, w = image.shape[0:2]
    src_points = np.array([[0, 0], [0, h], [w, 0], [w, h]], np.float32)
    for i in range (0, 4, 2):
        for j in range (0, 4, 2):
            if (i == 0 and j == 0):
                continue;
            dst_points = np.array([[0, 0], [0, h], [w, -i], [w, h+j]], np.float32)
            mat = cv.getPerspectiveTransform(src_points, dst_points)
            resimage = cv.warpPerspective(image, mat, (w, h))
            cv.imwrite("images_enlarge/p1" +str(i)+str(j) + name, resimage)
            dst_points = np.array([[-i, 0], [0, h], [w+j, 0], [w, h]], np.float32)
            mat = cv.getPerspectiveTransform(src_points, dst_points)
            resimage = cv.warpPerspective(image, mat, (w, h))
            cv.imwrite("images_enlarge/p2" + str(i) + str(j) + name, resimage)
            dst_points = np.array([[0, 0], [-i, h], [w, 0], [w+j, h]], np.float32)
            mat = cv.getPerspectiveTransform(src_points, dst_points)
            resimage = cv.warpPerspective(image, mat, (w, h))
            cv.imwrite("images_enlarge/p3" + str(i) + str(j) + name, resimage)
            dst_points = np.array([[0, -i], [0, h+j], [w, 0], [w, h]], np.float32)
            mat = cv.getPerspectiveTransform(src_points, dst_points)
            resimage = cv.warpPerspective(image, mat, (w, h))
            cv.imwrite("images_enlarge/p4" + str(i) + str(j) + name, resimage)
            
def blur(image, name):
    resimage = cv.GaussianBlur(image, (5, 5), 0)
    cv.imwrite("images_enlarge/b" + name, resimage)
    return resimage

list = os.listdir("images_each/")
#list = ["0_0.png"]
for elm in list:
    src = cv.imread("images_each/" + elm)
    cv.imwrite("images_enlarge/" + elm, src)
    cv.imwrite("images_enlarge/g" + elm, cv.cvtColor(src, cv.COLOR_RGB2GRAY))
    src1, src2 = move(blur(src, elm), "b" + elm)
    src3, src4 = move(src, elm)
    project(blur(src, elm), "b" + elm)
    project(src3, "m1"+elm)
    project(src4, "m2"+elm)
    project(src1, "m1b" + elm)
    project(src2, "m2b" + elm)
