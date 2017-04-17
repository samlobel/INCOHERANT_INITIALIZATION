import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# I don't know if it'll really be any better in a CNN actually, they have
# a crazy number of parameters already, meaning that they're probably pretty
# orthogonal. Bummer, I guess. 


##############################################################################################################
"""
Three step training program. First, you make the later layers 'orthogonal-ish'.
Then, you optimize the first layer to work well with them.
Then, you optimize everything!

Works well enough, not sure if the second part is strictly necessary.
"""


def _calculate_orthogonality_for_fc_layer(fc_tensor):
  """
  I've essentially done this before, for my dad. How did that work again?

  First, you're looking at things from one input to every output. That's a regular slice.
  So, tensor[0], tensor[1], tensor[2], etc.

  I think it might just be a matmul transpose(a). It might be super easy. Or, actually the other way.
  tf.matmul(tf.transpose(a), a). But then, you need to do the normalization part.
  You take the diagonal of that matrix, then you collapse it up to a flat guy, then you sqrt, then you outer product.

  Then, you have (a * b) / (||a|| ||b||)
  And that's what you get! I think scale is probably pretty important here though, cause these numbers could be
  outrageous. Only time will tell. 
  """
  # dots = tf.matmul(tf.transpose(fc_tensor), fc_tensor) # Correct dimensions, I think this is right.
  dots = tf.matmul(fc_tensor, tf.transpose(fc_tensor)) # ALTERNATIVE DIMENSIONS!!!

  diag_part = tf.diag_part(dots)
  norms = tf.expand_dims(tf.sqrt(diag_part), 0) # I think this is right, it gives you the norms...
  
  norms_mul = tf.matmul(tf.transpose(norms), norms) #Norms multiplied. That looks right to me. Don't switch this one, obviously!
  print(norms_mul.get_shape())

  scaled = tf.div(dots, norms_mul)
  scaled -= tf.diag(tf.diag_part(scaled)) # No need for the diagonals, which should all just be ones or something...
  return scaled #Maybe I should return just the size, but that's easy to calculate.
  # norms_mul = tfmatmul

  # pass

def _calculate_orthogonality_penalty(fc_tensor):
  """
  Square it, sum it up
  """
  orth_mat = _calculate_orthogonality_for_fc_layer(fc_tensor)
  # penalty = tf.reduce_mean(tf.square(orth_mat))
  penalty = tf.reduce_mean(tf.abs(orth_mat))
  return penalty


BATCH_SIZE = 1000
ORTHOG_DECAY = 1e2
# ORTHOG_DECAY = 1.0
WEIGHT_DECAY = 0.01



x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])



W1 = tf.get_variable("W1", shape=[784, 200],
           initializer=tf.contrib.layers.xavier_initializer())
           # tf.Variable(tf.random_uniform([784, 200], -0.1,0.1))
b1 = tf.Variable(tf.zeros([200]))

h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

W2 = tf.get_variable("W2", shape=[200, 100],
           initializer=tf.contrib.layers.xavier_initializer())

#tf.Variable(tf.random_uniform([1024, 1024], -0.1,0.1))
b2 = tf.Variable(tf.zeros([100]))

h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)


W3 = tf.get_variable("W3", shape=[100, 10],
           initializer=tf.contrib.layers.xavier_initializer())
    # W3 = tf.Variable(tf.random_uniform([200, 10], -0.1,0.1))
b3 = tf.Variable(tf.zeros([10]))

y_unscaled = tf.matmul(h2, W3) + b3
y = tf.nn.softmax(tf.matmul(h2, W3) + b3)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_unscaled, y_))

orths = [_calculate_orthogonality_penalty(tsr) for tsr in [W2, W3]] #Don't want the first one to be orthogonal, it's not its fault if it isn't

# last_orth = _calculate_orthogonality_for_fc_layer(W3)

loss_orth = tf.add_n(orths) * ORTHOG_DECAY

# total_error = cross_entropy + loss_orth
# total_error = loss_orth
total_error = cross_entropy


train_step_ortho = tf.train.MomentumOptimizer(1.0, 0.2).minimize(loss_orth)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# train_step = tf.train.MomentumOptimizer(0.01, 0.1).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(0.001).minimize(total_error)


first_layer_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(total_error, var_list=[W1])


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print('running')
for i in range(30000):
  if i % 50 == 0:
    print(i)
    # print(sess.run(last_orth))    
    correct_prediction = tf.equal(tf.argmax(y_unscaled,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print(sess.run([accuracy, cross_entropy, loss_orth, total_error, orths[-1]], feed_dict={x: mnist.test.images, y_: mnist.test.labels})[0:2])

  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



