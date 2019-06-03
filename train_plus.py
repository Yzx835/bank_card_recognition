import time
from datetime import datetime
import math
import tensorflow as tf
import numpy as np
import os
import cv2 as cv

"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32, name='b'))
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


def inference_op(input_op, keep_prob):
    p = []
    # assume input_op shape is 48 * 32 * 1

    # block 1 -- outputs 24 * 16 * 64
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name='pool1', kh=2, kw=2, dh=2, dw=2)

    # block 2 -- outputs 12 * 8 * 128
    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dh=2, dw=2)

    # block 3 -- outputs 6 * 4 * 256
    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dh=2, dw=2)

    # flatten
    shp = pool3.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool3, shape=[-1, flattened_shape], name='resh1')

    # fully connected
    fc1 = fc_op(resh1, name='fc1', n_out=1024, p=p)
    fc1_drop = tf.nn.dropout(fc1, keep_prob, name='fc1_drop')

    fc2 = fc_op(fc1_drop, name='fc2', n_out=1024, p=p)
    fc2_drop = tf.nn.dropout(fc2, keep_prob, name='fc2_drop')

    fc3 = fc_op(fc2_drop, name='fc3', n_out=10, p=p)
    softmax = tf.nn.softmax(fc3)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc3, p


def run_benchmark():
    with tf.Graph().as_default():
        x = tf.placeholder("float", shape=[None, image_height, image_width, channels])
        y_ = tf.placeholder("float", shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc3, p = inference_op(x, keep_prob)

        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)

        # time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")

        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(softmax, 1e-10, 1.0)))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(predictions, tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        sess.run(tf.global_variables_initializer())

        num_steps_burn_in = 0
        total_duration = 0.0
        total_duration_squared = 0.0

        for i in range(num_batches + num_steps_burn_in):
            batch = next_batch(batch_size, train_temp, train_np)
            feed = {x: batch[0], y_: batch[1], keep_prob: 0.5}
            feed_test = {x: batch[0].reshape([-1, image_height, image_width, channels]), y_: batch[1], keep_prob: 1.0}
            start_time = time.time()
            _ = sess.run(train_step, feed_dict=feed)
            duration = time.time() - start_time
            if i >= num_steps_burn_in:
                if not (i - num_steps_burn_in) % 10:
                    train_accuracy, _loss = sess.run((accuracy, cross_entropy), feed_dict=feed_test)
                    print('%s: step %d, duration = %.3f, loss = %.8f, training accuracy %g' % (datetime.now(), i - num_steps_burn_in, duration, _loss, train_accuracy))
                total_duration += duration
                total_duration_squared += duration * duration
        mn = total_duration / num_batches
        vr = total_duration_squared / num_batches - mn * mn
        sd = math.sqrt(vr)
        print('%s: across %d steps, %.3f +/- %.3f sec / batch' %
              (datetime.now(), num_batches, mn, sd))

        saver = tf.train.Saver()
        saver.save(sess, "train_plus-result/train_plus", global_step=num_batches)

        out_put = 0
        for i in range(len(test_np)):
            batch = next_batch(1, test_temp, test_np)
            feed_test_set = {x: batch[0].reshape([-1, image_height, image_width, channels]), y_: batch[1], keep_prob: 1.0}
            out_put += sess.run(accuracy.name, feed_dict=feed_test_set) / len(test_np)
        print(out_put)


def test_model():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('train_plus-result/train_plus-18605.meta')
    saver.restore(sess, tf.train.latest_checkpoint("train_plus-result/"))
    out_put = 0
    for i in range(len(test_np)):
        batch = next_batch(1, test_temp, test_np)
        feed_test_set = {'Placeholder:0': batch[0].reshape([-1, image_height, image_width, channels]), 'Placeholder_1:0': batch[1], 'Placeholder_2:0': 1.0}
        out_put += sess.run('Mean:0', feed_dict=feed_test_set) / len(test_np)
    print(out_put)


image_height = 48
image_width = 32
channels = 3
data_path = "data/"
train_np = np.load(data_path + "train.npy")
test_np = np.load(data_path + "test.npy")
train_temp = [0]
test_temp = [0]


def next_batch(count, temp, npy):
    np.random.shuffle(npy)
    size = len(npy)
    _temp = temp[0]
    assert(count <= size)
    img_list = []
    lab_list = []
    for i in range(count):
        img_list.append(npy[_temp][0])
        lab_list.append(npy[_temp][1])
        """
        cv.imshow("image", npy[_temp][0])
        print("num ; ", npy[_temp][1])
        cv.waitKey(0)
        """
        _temp = _temp + 1
        if _temp == size:
            np.random.shuffle(npy)
            _temp = 0
    temp[0] = _temp
    return np.array(img_list).reshape(-1, 48, 32, 3), np.array(lab_list)


batch_size = 20
epoch = 100
num_batches = int((epoch * len(train_np)) / batch_size)
print(num_batches)
if __name__ == '__main__':
    # run_benchmark()
    test_model()


