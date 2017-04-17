import tensorflow as tf
import numpy as np

import ops

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



BATCH_SIZE = 100
ORTHOG_DECAY = 1e2
# ORTHOG_DECAY = 1.0
WEIGHT_DECAY = 0.01


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

L1, _ = ops.fc_and_ortho_penalty(x, 'L1', [784, 200])

L2, ortho_penalty_2 = ops.fc_and_ortho_penalty(L1, 'L2', [200, 100])

L3, ortho_penalty_3 = ops.fc_and_ortho_penalty(L2, 'L3', [100, 10], act=tf.identity)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(L3, y_))

ORTHO_LOSS = ORTHOG_DECAY * (ortho_penalty_2 + ortho_penalty_3)

train_step_ortho = tf.train.MomentumOptimizer(0.01, 0.2).minimize(ORTHO_LOSS)

train_step = tf.train.MomentumOptimizer(0.01, 0.1).minimize(cross_entropy)




def train_orthos(sess, num_times=1000):
  for i in xrange(num_times):
    if i % 100 == 0:
      _ortho_loss = sess.run(ORTHO_LOSS)
      print('BATCH: {}\t\tORTHO LOSS: {}'.format(i, _ortho_loss))
    sess.run(train_step_ortho)

def train_on_mnist(sess, num_times=10000):
  for i in xrange(num_times):
    if i % 100 == 0:
      correct_prediction = tf.equal(tf.argmax(L3,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      
      _acc, _ce = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})[0:2]
      print('BATCH: {}\t\tACC: {}\t\tCE: {}'.format(i, _acc, _ce))

    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

train_orthos(sess, num_times=3000)
train_on_mnist(sess)







