import os
import random
import cv2 as cv
import numpy as np

src_path = "images_each/"
result_path = "data/"

train = []
test = []


def jpg_to_video(path, fps, data):
    # fourcc = cv.VideoWriter_fourcc(*"MJPG")
    # vw = cv.VideoWriter(path, fourcc, fps, (48, 32))
    count = 0
    for i in range(len(data)):
        lab_num = 0
        for j in range(10):
            if data[i][1][j] == 1:
                lab_num = j
                break
        cv.imwrite(path + str(count) + ".png", data[i][0])
        count += 1
    # vw.release()


def reshape_and_save(list, num, dst):
    for elm in list:
        img = cv.imread(src_path + elm)
        img = cv.resize(img, (32, 48))
        # cv.imshow("test", img)
        # cv.waitKey(0)
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lab = np.zeros(10, dtype=np.float)
        lab[int(num)] = 1.
        dst.append((img, lab))


nums = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "_": []}
list = os.listdir(src_path)
for file in list:
    nums[file[0]].append(file)

for num in nums.keys():
    if num is not "_":
        get_train = random.sample(nums[num], int(0.9 * len(nums[num])))
        for train_elm in get_train:
            nums[num].remove(train_elm)
        reshape_and_save(get_train, num, train)
        reshape_and_save(nums[num], num, test)
random.shuffle(train)
random.shuffle(test)
"""
train_list = []
train_label = []
test_list = []
test_label = []
print(len(train))
for i in range(len(train)):
    train_list.append(train[i][0])
    train_label.append(train[i][1])
for i in range(len(test)):
    test_list.append(test[i][0])
    test_label.append(test[i][1])

"""

jpg_to_video("./train_images/", 5, train)

np.save(result_path + "train.npy", np.array(train))
np.save(result_path + "test.npy", np.array(test))
