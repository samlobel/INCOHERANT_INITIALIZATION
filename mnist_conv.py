import tensorflow as tf
import numpy as np

import ops

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



BATCH_SIZE = 10
ORTHOG_DECAY = 1e2
# ORTHOG_DECAY = 1.0
WEIGHT_DECAY = 0.01
# MODEL = "1"
MODEL = "2"
print("MODEL: {}".format(MODEL))


x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28,28,1])

y_ = tf.placeholder(tf.float32, [None, 10])


if MODEL == "1":
  L1 = ops.conv_layer(x_image, 'CONV_1', [5, 5, 1, 32], downsample=True, norm=False)
  L2 = ops.conv_layer(L1, 'CONV_2', [5, 5, 32, 32], downsample=True, norm=False)

  fc_in = tf.reshape(L2, [-1, 7*7*32])

  FC_1, ortho_penalty_1 = ops.fc_and_ortho_penalty(fc_in, 'FC_1', [7*7*32, 384])
  FC_2, ortho_penalty_2 = ops.fc_and_ortho_penalty(FC_1, 'FC_2', [384, 100])
  FC_3, ortho_penalty_3 = ops.fc_and_ortho_penalty(FC_2, 'FC_3', [100, 10])

  out = FC_3
  ORTHO_LOSS = ORTHOG_DECAY * (ortho_penalty_1 + ortho_penalty_2 + ortho_penalty_3)

if MODEL == "2":
  L1 = ops.conv_layer(x_image, 'CONV_1', [5, 5, 1, 8], downsample=True, norm=False)

  fc_in = tf.reshape(L1, [-1, 14*14*8])

  FC_1, ortho_penalty_1 = ops.fc_and_ortho_penalty(fc_in, 'FC_1', [14*14*8, 384])
  FC_2, ortho_penalty_2 = ops.fc_and_ortho_penalty(FC_1, 'FC_2', [384, 10])

  out = FC_2
  ORTHO_LOSS = ORTHOG_DECAY * (ortho_penalty_1 + ortho_penalty_2)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y_))


train_step_ortho = tf.train.MomentumOptimizer(1.0, 0.2).minimize(ORTHO_LOSS)
train_step = tf.train.MomentumOptimizer(0.01, 0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      



def train_orthos(sess, num_times=1000):
  for i in xrange(num_times):
    if i % 5 == 0:
      _ortho_loss = sess.run(ORTHO_LOSS)
      print('BATCH: {}\t\tORTHO LOSS: {}'.format(i, _ortho_loss))
    sess.run(train_step_ortho)

def train_on_mnist(sess, num_times=10000):
  for i in xrange(num_times):
    if i % 100 == 0:
      _acc, _ce = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images[0:5000], y_: mnist.test.labels[0:5000]})
      print('BATCH: {}\t\tACC: {}\t\tCE: {}'.format(i, _acc, _ce))

    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print('initialized!')

train_orthos(sess, num_times=100)
train_on_mnist(sess)



