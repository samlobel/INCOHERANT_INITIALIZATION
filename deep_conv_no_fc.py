"""
Copy the stuff from mnist_conv into here, but use the pre-init conv.
"""

import tensorflow as tf
import numpy as np

import ops

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



BATCH_SIZE = 50
# ORTHOG_DECAY = 1e2
# ORTHOG_DECAY = 1.0
WEIGHT_DECAY = 0.01
# MODEL = "1"


x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28,28,1])

y_ = tf.placeholder(tf.float32, [None, 10])


# ITERS = 2000
ITERS=0
ITERS_BIG=50
# ITERS_BIG=0

print('downsampling in the beginning!')
out = ops.conv_layer(x_image, 'CONV_1', [5, 5, 1, 10], act=tf.nn.elu, downsample=True, norm=False)
for i in range(2,7):
  out = ops.special_ortho_conv_layer(out, 'CONV_{}'.format(i), [5, 5, 10, 10], act=tf.nn.elu, downsample=False, norm=False)
out = ops.conv_layer(out, 'CONV_7', [5, 5, 10, 10], act=tf.identity, downsample=False, norm=False)

out = tf.reduce_mean(out, reduction_indices=[1,2])
print('shape of out: {}'.format(out.get_shape()))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y_))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



NUM_IMAGES_TO_TEST_ON=5000
def train_on_mnist(sess, num_times=10000):
  for i in xrange(num_times):
    if i % 10 == 0:
      _acc, _ce = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images[0:NUM_IMAGES_TO_TEST_ON], y_: mnist.test.labels[0:NUM_IMAGES_TO_TEST_ON]})
      print('BATCH: {}\t\tACC: {}\t\tCE: {}'.format(i, _acc, _ce))

    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print('initialized!')

# train_orthos(sess, num_times=100)
train_on_mnist(sess)
