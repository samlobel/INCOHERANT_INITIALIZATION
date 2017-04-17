import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_uniform([784, 100], -0.1,0.1))
b1 = tf.Variable(tf.zeros([100]))

h1 = tf.nn.tanh(tf.matmul(x,W1) + b1)

W2 = tf.Variable(tf.random_uniform([100, 100], -0.1,0.1))
b2 = tf.Variable(tf.zeros([100]))

h2 = tf.nn.tanh(tf.matmul(h1,W2) + b2)

W3 = tf.Variable(tf.random_uniform([100, 10], -0.1,0.1))
b3 = tf.Variable(tf.zeros([10]))


y = tf.nn.softmax(tf.matmul(h2, W3) + b3)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print('running')
for i in range(10000):
	if i % 100 == 0:
		print(i)
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
