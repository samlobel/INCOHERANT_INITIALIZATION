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
MODEL = "1"
print("MODEL: {}".format(MODEL))


x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28,28,1])

y_ = tf.placeholder(tf.float32, [None, 10])


# ITERS = 2000
ITERS=0
ITERS_BIG=50
# ITERS_BIG=0

if MODEL == "1":
  L1 = ops._get_ortho_init_conv_layer(x_image, 'CONV_1', [3, 3, 1, 64], act=tf.nn.elu, downsample=True, num_iters=0, norm=False)
  L2 = ops.special_ortho_conv_layer(L1, 'CONV_2', [3, 3, 64, 64], act=tf.nn.elu, lr=10, num_iters=0, downsample=True, norm=False)

  fc_in = tf.reshape(L2, [-1, 7*7*64])
  FC_1 = ops._get_ortho_init_fc_layer(fc_in, [7*7*64, 384], 'FC_1', lr=50, num_iters=0, act=tf.nn.elu, bias=True)
  FC_2 = ops._get_ortho_init_fc_layer(FC_1, [384, 100], 'FC_2', lr=50, num_iters=0, act=tf.nn.elu, bias=True)
  FC_3 = ops._get_ortho_init_fc_layer(FC_2, [100, 10], 'FC_3', lr=5, num_iters=0, act=tf.identity, bias=True)

  # FC_1, ortho_penalty_1 = ops.fc_and_ortho_penalty(fc_in, 'FC_1', [7*7*32, 384])
  # FC_2, ortho_penalty_2 = ops.fc_and_ortho_penalty(FC_1, 'FC_2', [384, 100])
  # FC_3, ortho_penalty_3 = ops.fc_and_ortho_penalty(FC_2, 'FC_3', [100, 10])

  out = FC_3
  # ORTHO_LOSS = ORTHOG_DECAY * (ortho_penalty_1 + ortho_penalty_2 + ortho_penalty_3)

if MODEL == "2":
  L1 = ops.conv_layer(x_image, 'CONV_1', [5, 5, 1, 8], downsample=True, norm=False)

  fc_in = tf.reshape(L1, [-1, 14*14*8])

  FC_1, ortho_penalty_1 = ops.fc_and_ortho_penalty(fc_in, 'FC_1', [14*14*8, 384])
  FC_2, ortho_penalty_2 = ops.fc_and_ortho_penalty(FC_1, 'FC_2', [384, 10])

  out = FC_2
  ORTHO_LOSS = ORTHOG_DECAY * (ortho_penalty_1 + ortho_penalty_2)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y_))

# train_step_ortho = tf.train.MomentumOptimizer(1.0, 0.2).minimize(ORTHO_LOSS)
# train_step = tf.train.MomentumOptimizer(0.01, 0.1).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# def train_orthos(sess, num_times=1000):
#   for i in xrange(num_times):
#     if i % 5 == 0:
#       _ortho_loss = sess.run(ORTHO_LOSS)
#       print('BATCH: {}\t\tORTHO LOSS: {}'.format(i, _ortho_loss))
#     sess.run(train_step_ortho)


NUM_IMAGES_TO_TEST_ON=5000
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
