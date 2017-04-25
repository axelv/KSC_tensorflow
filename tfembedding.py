import tensorflow as tf
import numpy as np
import os

os.getcwd()
LOG_DIR = 'embeddings/'

fname = "embedding.npy"
npa = np.load(fname)
data = tf.Variable(npa, name="emb_data")
#np.save("frigo_autoenc", npa)
with tf.Session() as sess:
    saver = tf.train.Saver([data])
    sess.run(data.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))
npa.shape
