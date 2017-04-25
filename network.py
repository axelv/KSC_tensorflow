import tensorflow as tf
import numpy as np
from layers import *
import io_tools
import plot_tools
from KSC import KSC

# GENERAL PARAMETERS

k = 6 # num of clusters
d = 5 # input dimension for the clustering
n_train = 100


def encoder():
    pass

def decoder():
    pass

def ksc_net(x, ksc_model: KSC):
    with tf.variable_scope('Cluster Net'):
        kernel_x = RBF_layer(x, [n_train, d], ksc_model.get_centroids_initializer(), ksc_model.get_sigma_initializer())

        score_x = linear_layer(kernel_x, [k-1, n_train], ksc_model.get_alpha_initializer(), ksc_model.get_bias_initializer())

        cosd_x = cosine_layer(score_x, k, ksc_model.get_prototype_initializer())

    return cosd_x

