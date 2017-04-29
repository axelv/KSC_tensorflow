import numpy as np
import tensorflow as tf

np_tensor = np.concatenate([np.eye(4)[:, :, None], 2*np.eye(4)[:, :, None], 3*np.eye(4)[:, :, None], 4*np.eye(4)[:, :, None]], axis=2)

constant = tf.constant(np_tensor)

result = tf.matrix_set_diag(constant, np.ones([4, 4]))

with tf.Session() as sess:
    print(constant)
    print(sess.run(result))
