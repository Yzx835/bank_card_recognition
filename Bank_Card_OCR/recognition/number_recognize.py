import sys
import os
sys.path.append("../")
import tensorflow as tf
import cv2 as cv
from data.constants import *

"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""

result_path = RESULT_MODEL_RESULT_PATH
model_name = MODEL_NAME

image_height = 48
image_width = 32
channels = 3
classes = 11

sess = tf.Session()
saver = tf.train.import_meta_graph(result_path + model_name + ".meta")
saver.restore(sess, tf.train.latest_checkpoint(result_path))


def number_recognize(src_image):
    src_image = cv.resize(src_image, (32, 48))
    feed_test_set = {'Placeholder:0': src_image.reshape([-1, image_height, image_width, channels]),
                     'Placeholder_2:0': 1.0}
    out_put = sess.run('ArgMax:0', feed_dict=feed_test_set)
    out_put = out_put[0]
    if out_put == 10:
        out_put = "_"
    return str(out_put)


"""
if __name__ == '__main__':
    test_path = "test_images/"
    file_list = os.listdir(test_path)
    for file in file_list:
        image = cv.imread(test_path + file)
        print(file + " : " + number_recognize(image))
"""