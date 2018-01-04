import os
import io
import shutil
import tensorflow as tf
import tensorflow.contrib.tensorboard.plugins.projector as projector
import numpy as np
from layers import *
import io_tools
import plot_tools as plt
from KSC import KSC
from sklearn.datasets import make_blobs

# DESCRIPTION
# Demo of a network version of KSC in 'out-of-sample' mode

# GENERAL PARAMETERS

k = 3  # num of clusters
d = 2
n_train = 500
batch_size = 400
n_batch = 100
n_test = batch_size * n_batch
centers = [(-5, -5), (0, 0), (5, 5)]
x_offset = np.concatenate([np.linspace(0, 5, n_batch/2)[:, None], np.linspace(0, -5, n_batch/2)[:, None]], axis=1)
x_offset = np.concatenate([x_offset, np.array([5, -5])*np.ones([n_batch, 2])], axis=0)
eps = np.finfo(float).eps
learning_rate=0.05

LOG_DIR = "logs/KSC_dyn_v0.3/eta_"+str(learning_rate)

# Create new run
i = 0
while os.path.exists(LOG_DIR+"/"+str(i)):
    i = i + 1
LOG_DIR_X = LOG_DIR+"/"+str(i)
os.makedirs(LOG_DIR_X)


def encoder():
    pass


def decoder():
    pass

def write_image(image, name="Plot"):

    return tf.summary.image(name, image)

def cross_entropy(y, y_true):
    with tf.name_scope('Cross_entropy'):
        ce = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y + eps), axis=1))
        tf.summary.scalar('cross entropy', ce)
    return ce


def train(loss, global_step, learning_rate=0.05):
    with tf.name_scope('Train'):
        my_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_step = my_opt.minimize(loss, global_step=global_step)

    return train_step

def create_id_translator(y_truth, y_data):
    """
    Creates translation vector that translates the dataset indicators to the KSC indicators
    y[i_dataset] = i_KSC
    
    :param y_truth: 
    :param y_data: 
    :return: 
    """
    cluster_ids, indices = np.unique(y_data, return_index=True)
    y_trans = np.zeros_like(cluster_ids)
    y_trans[y_truth[indices]] = cluster_ids

    return y_trans


def hot_encoder(a):

    depth = (a.max()-a.min())+1
    length = a.shape[0]
    b = np.zeros((length, depth))
    b[np.arange(length), a] = 1

    return b


def ksc_net(x, ksc_model: KSC, plot=True):
    with tf.variable_scope('cluster_net'):
        kernel_x = RBF_layer(x, [n_train, d], ksc_model.get_centroids_initializer(), ksc_model.get_sigma_initializer())

        score_x = linear_layer(kernel_x, [n_train, k - 1], ksc_model.get_alpha_initializer(),
                               ksc_model.get_bias_initializer())

        cosd_x = cosine_layer(score_x, k, ksc_model.get_prototype_initializer())

        cm_x = cluster_membership_layer(cosd_x)

        # Summaries
        tf.summary.histogram("cluster membership", cm_x)

    return cm_x, score_x


def get_data():
    x_data, y_data = make_blobs(n_samples=n_train + n_test, n_features=d, cluster_std=0.5, centers=centers)
    X_train = x_data[0:n_train]
    y_train = y_data[0:n_train]
    X_test = x_data[n_train:n_train + n_test]
    y_test = y_data[n_train:n_train + n_test]

    return X_train, y_train, X_test, y_test


with tf.Graph().as_default():
    with tf.variable_scope("ksc_net"):
        sigma_float = 0.25
        k_int = 3
        sigma = np.array([sigma_float])
        k = np.array([k_int])

        X_train, y_train, X_test, y_test = get_data()

        x = tf.placeholder(tf.float64, [batch_size, d], name="x")
        y = tf.placeholder(tf.float64, [batch_size, k], name="y")
        img = tf.placeholder(tf.uint8, name="img")

        # Construction & pre-training of KSC-model
        ksc_model = KSC(X_train, k, sigma)
        # PLOT pre-train
        plt.plot2D(X_train, y=ksc_model.cm, title="KSC train data")

        # TF Ops
        global_step = tf.Variable(0, name='global_step', trainable=False)
        cluster_op, score_op = ksc_net(x, ksc_model)
        ce_op = cross_entropy(cluster_op, y)
        train_op = train(ce_op, global_step, learning_rate=learning_rate)
        init_op = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()
        img_op = write_image(img, "prediction")

        # Setup TensorFlow
        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(init_op)
        writer = tf.summary.FileWriter(LOG_DIR_X, graph=sess.graph)

        np.savetxt(LOG_DIR_X+"/metadata.tsv", y_train.round().astype('int32'), delimiter='\t')

        score_train = sess.run(score_op, feed_dict={x: X_train[0:batch_size]})
        cm_pre_train = sess.run(cluster_op, feed_dict={x: X_train[0:batch_size]})
        q_pre_train = np.argmax(cm_pre_train, axis=1)

        # TODO: possibly not evey cluster is represented in the first batch
        y_trans = create_id_translator(y_train[0:batch_size], q_pre_train)

        for i in range(n_batch):

            y_batch = hot_encoder(y_trans[y_test[i * batch_size: (i + 1) * batch_size]])
            x_batch = X_test[i * batch_size:(i + 1) * batch_size]
            cluster_indicators = y_test[i * batch_size: (i + 1) * batch_size]
            x_batch[cluster_indicators == 0, :] = x_batch[cluster_indicators == 0, :] + x_offset[i,:]

            [summary, train_pred, train_score, train_ce, _] = sess.run([summary_op, cluster_op, score_op,
                                                                        ce_op, train_op],
                                                                       feed_dict={x: x_batch, y: y_batch})

            saver.save(sess, os.path.join(LOG_DIR_X, "model.ckpt"), global_step)

            fig = plt.plot2D(x_batch, y=train_pred.round().astype('int32'), title="Batch "+str(i), axes=[-10, 10, -15, 10])
            img_data = plt.process_plot(fig)

            # Plots
            writer.add_summary(sess.run(img_op, feed_dict={img: img_data}), sess.run(global_step))
            writer.add_summary(summary, sess.run(global_step))

        writer.close()
