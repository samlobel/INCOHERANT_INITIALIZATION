import tensorflow as tf
import numpy as np
import ops



def orthonormal(shape):
  rand_mat = np.random.rand(*shape)
  orth = np.linalg.qr(rand_mat)[0]
  return orth


def test_ortho_init(shape):
  ortho = orthonormal(shape)
  ortho_tensor = tf.convert_to_tensor(ortho)
  penalty = ops._calculate_orthogonality_penalty(ortho_tensor)
  with tf.Session() as sess:
    print(penalty.eval())
  pass

def test_optimizing_ortho_init(shape):
  ortho = np.multiply(orthonormal(shape), 10)
  print(ortho.shape)
  ortho_v = tf.Variable(tf.convert_to_tensor(ortho))
  penalty = ops._calculate_orthogonality_penalty(ortho_v)
  opt = tf.train.GradientDescentOptimizer(100.).minimize(penalty)
  init = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init)
    for i in xrange(10000):
      if i % 100 == 0:
        print('batch: {}\t\tpenalty: {}'.format(i, sess.run(penalty)))
      sess.run(opt)

def test_ortho_init(shape):
  mat = ops._get_ortho_init_fc_layer(shape, "Who_Cares", num_iters=2500, lr=5)
  




if __name__ == '__main__':
  # orth = orthonormal((3,2))
  # print(orth)
  # test_ortho_init((100,10))
  # test_optimizing_ortho_init((200,100))
  test_ortho_init((200, 100))


  

