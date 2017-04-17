"""
Copy the stuff from mnist_conv into here, but use the pre-init conv.
"""

import tensorflow as tf
import numpy as np

import ops

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



BATCH_SIZE = 2000
# ORTHOG_DECAY = 1e2
# ORTHOG_DECAY = 1.0
WEIGHT_DECAY = 0.01
# MODEL = "1"
MODEL = "1"
print("MODEL: {}".format(MODEL))


x = tf.placeholder(tf.float32, [None, 784])
# x_image = tf.reshape(x, [-1, 28,28,1])

y_ = tf.placeholder(tf.float32, [None, 10])


# ITERS = 2000
ITERS=0
ITERS_BIG=50
# ITERS_BIG=0

if MODEL == "1":
  FC_1 = ops._get_ortho_init_fc_layer(x, [784, 1000], 'FC_1', lr=0, num_iters=0, act=tf.nn.elu, bias=True)
  FC_2 = ops._get_ortho_init_fc_layer(FC_1, [1000, 500], 'FC_2', lr=10, num_iters=0, act=tf.nn.elu, bias=True)
  FC_3 = ops._get_ortho_init_fc_layer(FC_2, [500, 250], 'FC_3', lr=10, num_iters=0, act=tf.nn.elu, bias=True)
  FC_4 = ops._get_ortho_init_fc_layer(FC_3, [250, 125], 'FC_4', lr=10, num_iters=0, act=tf.nn.elu, bias=True)
  FC_5 = ops._get_ortho_init_fc_layer(FC_4, [125, 64], 'FC_5', lr=10, num_iters=0, act=tf.nn.elu, bias=True)
  FC_6 = ops._get_ortho_init_fc_layer(FC_5, [64, 10], 'FC_6', lr=10, num_iters=0, act=tf.identity, bias=True)

  out = FC_6
  # ORTHO_LOSS = ORTHOG_DECAY * (ortho_penalty_1 + ortho_penalty_2 + ortho_penalty_3)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y_))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# def train_orthos(sess, num_times=1000):
#   for i in xrange(num_times):
#     if i % 5 == 0:
#       _ortho_loss = sess.run(ORTHO_LOSS)
#       print('BATCH: {}\t\tORTHO LOSS: {}'.format(i, _ortho_loss))
#     sess.run(train_step_ortho)


NUM_IMAGES_TO_TEST_ON=10000
def train_on_mnist(sess, num_times=10000):
  for i in xrange(num_times):
    if i % 25 == 0:
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
