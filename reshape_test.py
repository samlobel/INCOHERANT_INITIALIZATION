import numpy as np


# The question is, how does the multiplication happen? It's so confusiong!

# Be 2x2x1x3
conv_filter = [
  [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]],
  [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]
]
first_shape = [2,2,2,3]
second_shape = [8,3]

conv_filter = [
  [[[1,2,3]],[[1,2,3]]],
  [[[1,2,3]],[[1,2,3]]]
]
first_shape = [2,2,1,3]
second_shape = [4,3]


filter_np = np.asarray(conv_filter)
reshaped = np.reshape(filter_np, second_shape)
reshaped_T = reshaped.T

print('shape is {}'.format(filter_np.shape))
print('shape of reshaped: {}'.format(reshaped.shape))

# print(reshaped)
re_reshaped = np.reshape(reshaped, first_shape)
print('first: ')
print(filter_np)
print('reshaped:')
print(reshaped)
print('reshaped_T:')
print(reshaped_T)
print('rereshaped: ')
print(re_reshaped)

# print(np.reshape)



"""

Alright, conclusions:

The thing I made above is pretty illustrative. All the ones map to the first dimension, 
all the twos map to the second dimension, and all the threes map to the third dimension.

So, what's the deal? I think maybe the original optimziation scheme is correct.

Which sort of stinks, because it didn't give very good results :(

"""



"""
What for CNNs?

I think the train of thought that I'm going down is that, for a 5x5x64x64,
it is (5*5*64, 64), which is pretty much always orthogonal. So, 
orthogonalizing probably doesn't do much good. BUT, and I don't have the
best reasons for this, but I think that where you are matters. Something
like the center of one is the edge of another. For some reason, I think that
having each location be orthogonal to the other locations makes a lot of sense.

So, for that example above, I need to make 25 64x64 orthogonal matrices,
and somehow reshape it correctly... And I need the proper standard deviation!

If it was 1 to 32, I don't really know what to do. If it was 32 to 1,
I also don't really know what to do. 


"""

def new_cnn_init():
  shape = [3,3,8,8]
  mats = []
  x = (6.0 / (8.0+8.0))**0.5
  print(x)
  for i in xrange(9):
    rand = np.random.uniform(low=-x, high=x, size=(8,8))
    # # print(rand)
    # print(np.std(rand))
    orth = np.linalg.qr(rand)[0]
    # print(orth.dot(orth.T))
    mats.append(orth)
  mats = np.asarray(mats)
  print('mats shape: {}'.format(mats.shape))
  print('mean: {}\t\tstddev: {}'.format(np.mean(mats), np.std(mats)))
  print('range: {}'.format(x))
  mats_reshaped = np.reshape(mats(3,3,8,8))



new_cnn_init()


