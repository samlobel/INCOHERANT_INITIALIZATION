import tensorflow as tf
import numpy as np



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
  # penalty = tf.reduce_mean(orth_mat)
  return penalty


# def _get_ortho_init_fc_layer(shape, name, lr=1.0, num_iters=1000):
#   W = tf.get_variable(name + "_TEMP", shape=shape,
#            initializer=tf.contrib.layers.xavier_initializer())
#   penalty = _calculate_orthogonality_penalty(W)
#   opt = tf.train.GradientDescentOptimizer(lr).minimize(penalty)

#   val = None
#   with tf.Session() as sess:
#     sess.run(W.initializer)
#     for i in xrange(num_iters):
#       if i % 50 == 0:
#         print('batch: {}\t\tpenalty: {}'.format(i, sess.run(penalty)))
#       sess.run(opt)
#     val = sess.run(W)

#   to_return = tf.Variable(val, name=name)
#   return to_return

def _optimize_and_get_value(two_d_matrix, lr, num_iters):
  penalty = _calculate_orthogonality_penalty(two_d_matrix)
  opt = tf.train.GradientDescentOptimizer(lr).minimize(penalty)
  val = None
  
  print_on_mod = num_iters // 50 if num_iters >= 50 else 1
  print('printing every {} steps'.format(print_on_mod))

  with tf.Session() as sess:
    print('Starting optimization session')
    sess.run(two_d_matrix.initializer)
    for i in xrange(num_iters):
      if i % print_on_mod == 0:
        print('batch: {}\t\tpenalty: {}'.format(i, sess.run(penalty)))
      sess.run(opt)
    val = sess.run(two_d_matrix)
    print('optimization complete')
    print('val mean: {}'.format(np.mean(val)))
    print('val stddev: {}'.format(np.std(val)))

  return val

def _get_ortho_init_fc_matrix(shape, name, lr=1.0, num_iters=1000):
  W = tf.get_variable(name + "_TEMP_FOR_INIT", shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())
  init_val = _optimize_and_get_value(W, lr, num_iters)

  to_return = tf.Variable(init_val, name=name)
  return to_return

def _get_ortho_init_fc_layer(layer_in, shape, name, lr=1.0, num_iters=1000, act=tf.nn.relu, bias=True):
  init_matrix = _get_ortho_init_fc_matrix(shape, name, lr, num_iters)
  b = tf.Variable(tf.zeros([shape[-1]])) if bias else 0
  out = act(tf.matmul(layer_in, init_matrix) + b)
  return out


def _get_ortho_init_conv_filter(shape, name, lr=1.0, num_iters=1000):
  # Shape: filter shape. Goes, [f_height, f_width, in_channels, out_channels]
  if len(shape) != 4:
    print('shape: {}'.format(shape))
    raise Exception("Bad shape")
  new_shape = [shape[0]*shape[1]*shape[2] , shape[3]]
  fc_tensor = _get_ortho_init_fc_matrix(new_shape, name + "_FLAT", lr, num_iters)
  to_return = tf.reshape(fc_tensor, shape, name=name)
  return to_return

def _get_ortho_init_conv_layer(layer_in, name, shape, lr=1.0, num_iters=1000, act=tf.nn.relu, bias=True, downsample=False, norm=True):
  init_filter = _get_ortho_init_conv_filter(shape, name, lr, num_iters)
  conv = tf.nn.conv2d(layer_in, init_filter, [1, 1, 1, 1], padding='SAME')

  b = tf.Variable(tf.zeros([shape[-1]])) if bias else 0
  out = act(conv + b)
  if norm == True:
    out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                  name='norm2')
  if downsample == True:
    out = tf.nn.max_pool(out, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  return out





def pre_init_conv_layer(layer_in, name, shape, lr=1.0, num_iters=1000, act=tf.nn.relu, bias=True, downsample=False, norm=True):
  with tf.variable_scope(name) as scope:
    kernel = _get_ortho_init_conv_layer(shape, name, num_iters=100)
    # kernel = tf.get_variable(name, shape=shape,
    #             initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(layer_in, kernel, [1, 1, 1, 1], padding='SAME')
    b = tf.Variable(tf.zeros([shape[-1]])) if bias else 0
    out = act(conv + b)
    if norm == True:
      out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    if downsample == True:
      out = tf.nn.max_pool(out, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    return out


def ortho_filter_np(shape):
  mats = []
  if len(shape) != 4:
    raise Exception('bad shape for ortho filter: {}'.format(shape))
  if shape[2] != shape[3]:
    raise Exception('can only do same in and out channels, for now. {}'.format(shape))
  x = (6.0 / ((shape[0]*shape[1]*shape[2]) + shape[3] + 0.0))**0.5
  print(x)
  for i in xrange(shape[0]*shape[1]):
    rand = np.random.uniform(low=-x, high=x, size=shape[2:])
    # # print(rand)
    # print(np.std(rand))
    orth = np.linalg.qr(rand)[0]
    # print(orth.dot(orth.T))
    mats.append(orth)
  mats = np.asarray(mats)
  print('mats shape: {}'.format(mats.shape))
  print('mean: {}\t\tstddev: {}'.format(np.mean(mats), np.std(matsls)))
  print('range: {}'.format(x))
  mats_reshaped = np.reshape(mats, shape)
  return mats_reshaped



def special_ortho_conv_layer(layer_in, name, shape, lr=1.0, num_iters=1000, act=tf.nn.relu, bias=True, downsample=False, norm=True):
  init_filter = ortho_filter_np(shape)
  conv = tf.nn.conv2d(layer_in, init_filter, [1, 1, 1, 1], padding='SAME')

  b = tf.Variable(tf.zeros([shape[-1]])) if bias else 0
  out = act(conv + b)
  if norm == True:
    out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                  name='norm2')
  if downsample == True:
    out = tf.nn.max_pool(out, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  return out








def fc_and_ortho_penalty(layer_in, name, shape, act=tf.nn.relu, bias=True):
  with tf.variable_scope(name) as scope:
    W = tf.get_variable(name, shape=shape,
             initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros([shape[-1]])) if bias else 0

    layer = act(tf.matmul(layer_in, W) + b)
    ortho_penalty = _calculate_orthogonality_penalty(W)
    return layer, ortho_penalty

def conv_layer(layer_in, name, shape, act=tf.nn.relu, bias=True, downsample=False, norm=True):
  with tf.variable_scope(name) as scope:
    kernel = tf.get_variable(name, shape=shape,
                initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(layer_in, kernel, [1, 1, 1, 1], padding='SAME')
    b = tf.Variable(tf.zeros([shape[-1]])) if bias else 0
    out = act(conv + b)
    if norm == True:
      out = tf.nn.lrn(out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    if downsample == True:
      out = tf.nn.max_pool(out, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    return out





