import tensorflow as tf
import numpy as np
import os
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""

data_path = "data/"
train_np = np.load(data_path + "train.npy")
train_lab_np = np.load(data_path + "train_label.npy")
test_img_np = np.load(data_path + "test_image.npy")
test_lab_np = np.load(data_path + "test_label.npy")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x_conv, W_conv):
    return tf.nn.conv2d(x_conv, W_conv, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x_conv):
    return tf.nn.max_pool(x_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


temp = 0
size = train_img_np.shape[0]


def next_batch(count):
    global temp
    assert(count < size)
    img_list = []
    lab_list = []
    for i in range(count):
        img_list.append(train_img_np[temp].reshape(48 * 32))
        lab_list.append(train_lab_np[temp])
        temp = temp + 1
        if temp == size:
            temp = 0
    return np.array(img_list), np.array(lab_list)


x = tf.placeholder("float", shape=[None, 1536])
y_ = tf.placeholder("float", shape=[None, 10])
sess = tf.InteractiveSession()

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 48, 32, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


W_fc1 = weight_variable([6 * 4 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 6 * 4 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training acc uracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

saver = tf.train.Saver()
saver.save(sess, "train_plus-result/train_plus", global_step=20000)
"""
out_accuracy = 0.0
for i in range(1000):
    batch = mnist.test.next_batch(10)
    out_accuracy += 0.001 * accuracy.eval(feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 1.0
    })
print(out_accuracy)
"""

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: test_img_np.reshape((-1, 48 * 32)), y_: test_lab_np, keep_prob: 1.0}))
