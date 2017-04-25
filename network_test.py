import tensorflow as tf
import numpy as np
from layers import *
import io_tools
import plot_tools
from KSC import KSC
from sklearn.datasets import make_circles

# GENERAL PARAMETERS

k = 2 # num of clusters
d = 2 # input dimension for the clustering
n_train = 400
batch_size = 100
n_test = 100


def encoder():
    pass

def decoder():
    pass

def ksc_net(x, ksc_model: KSC):
    with tf.variable_scope('cluster_net'):
        kernel_x = RBF_layer(x, [n_train, d], ksc_model.get_centroids_initializer(), ksc_model.get_sigma_initializer())

        score_x = linear_layer(kernel_x, [n_train, k-1], ksc_model.get_alpha_initializer(), ksc_model.get_bias_initializer())

        cosd_x = cosine_layer(score_x, k, ksc_model.get_prototype_initializer())

    return cosd_x


with tf.Graph().as_default():
    with tf.variable_scope("ksc_net"):

        sigma_float = 0.1
        k_int = 2
        sigma = np.array([sigma_float])
        k = np.array([k_int])

        X_train, y = make_circles(n_samples=n_train, factor=0.3, noise=.05)
        X_test, y = make_circles(n_samples=n_test, factor=0.3, noise=.05)

        ksc_model = KSC(X_train, k, sigma)

        x = tf.placeholder(tf.float64, [None, d])
        #y = tf.placeholder(tf.float64, [None, k])

        cluster_out = ksc_net(x, ksc_model)

        init_op = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init_op)

        cosd_test = sess.run([cluster_out], feed_dict={x: X_test})


