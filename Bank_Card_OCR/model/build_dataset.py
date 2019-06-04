import os
import random
import cv2 as cv
import numpy as np
from data.constants import *

# DATA_PROCESS_EACH_PATH = "images_each/"
# DATA_PROCESS_ENLARGE_PATH = "images_enlarge/"
# MODEL_DATASET_PATH = "data/"

train = []
test = []


def reshape_and_save(file_list, label, dst):
    for elm in file_list:
        img = cv.imread(elm)
        img = cv.resize(img, (32, 48))
        # cv.imshow("test", img)
        # cv.waitKey(0)
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lab = np.zeros(11, dtype=np.float)
        if label == '_':
            lab[10] = 1.
        else:
            lab[int(label)] = 1.
        dst.append((img, lab))


def build_dataset():
    nums = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "_": []}
    list = os.listdir(DATA_PROCESS_EACH_PATH)
    for file in list:
        nums[file[0]].append(DATA_PROCESS_EACH_PATH + file)
    list = os.listdir(DATA_PROCESS_ENLARGE_PATH)
    for file in list:
        nums[file[0]].append(DATA_PROCESS_ENLARGE_PATH + file)

    for num in nums.keys():
        get_train = random.sample(nums[num], int(1.0 * len(nums[num])))
        for train_elm in get_train:
            nums[num].remove(train_elm)
        reshape_and_save(get_train, num, train)
        reshape_and_save(nums[num], num, test)

    np.save(MODEL_DATASET_PATH + "train.npy", np.array(train))
    np.save(MODEL_DATASET_PATH + "test.npy", np.array(test))


if __name__ == '__main__':
    build_dataset()
