import tensorflow as tf
import numpy as np
import scipy.spatial.distance as spdist
import numpy.linalg as nplin


def RBF_layer(x, centres_shape, centres_init=tf.random_normal_initializer(), sigma_init=tf.random_normal_initializer(),
              name="RBF_layer", reuse=None):
    """
    RBF (Radial Basis Function) Neural Network Layer.
    :param x: input vector
    :type x: ndarray
    :param centres_shape: Matrix with rows representing the centres of the RBF-kernels
    :type  centres_shape: ndarray
    """
    with tf.variable_scope(name, reuse=reuse):
        d_input = centres_shape[1]
        n_centres = centres_shape[0]

        centres = tf.get_variable("centres", [n_centres, d_input], initializer=centres_init)
        sigma = tf.get_variable("sigma", [d_input, d_input], initializer=sigma_init)

        # operation body
        x_ext = tf.tile(tf.expand_dims(x, 0), [n_centres, 1])

        # d² = (x-c)^T*SIGA^-1*(x-c)
        sigma_inv = tf.matrix_inverse(sigma)
        diff = x_ext - centres
        weighted_diff = tf.matmul(diff, sigma_inv)
        sqrd_dist_c = tf.reduce_sum(tf.square(weighted_diff), 1)
        y = tf.exp(-0.5 * sqrd_dist_c)

    return y


def linear_layer(x, size, weights_init=tf.random_normal_initializer(), biases_init=tf.random_normal_initializer(),
                 name="Linear_layer", reuse=None):
    """
    Linear Neural Network Layer
    """
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable("centres", size, initializer=weights_init)
        biases = tf.get_variable("biases", size[0], initializer=biases_init)

        y = tf.matmul(weights, x) + biases

    return y


def cosine_layer(x, size, weights_init=tf.random_normal_initializer(), name="Cosine_layer", reuse=None):
    """
    Cosine Distance Layer
    """
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable("centres", size, initializer=weights_init)
        y = tf.matmul(weights, x)
        y = y / tf.norm(y)

    return y


def construct_ksc(x_train, k_range, sigma_range):
    """
    Construct KSC model and return initializers for RBF Network
    :param x: Matrix with training data
    :type x: ndarray
    """
    n_train = x_train.shape[0]
    d_input = x_train.shape[1]
    l = sigma_range.shape[0]
    m = k_range.shape[0]

    # calculate initial kernel
    for i in range(l):
        sigma_sq_prev = sigma_sq
        sigma_sq = np.square(sigma_range[i, :, :])

        # update omega or calculate from scratch
        if (i == 0):
            omega = kernel_matrix(x_train, sigma_sq)
            # omega = 0.5*(omega + omega')
        else:
            omega = np.exp(np.matmul(np.inverse(sigma_sq), np.matmul(sigma_sq_prev * log(x_train))))

            # degree matrix
            D = np.diag(np.sum(omega, 1))
            D_inv = np.linalg.inv(D)

            MD = np.eye(N) - np.matmul(np.ones(n_train, d_input) * D_inv) / np.sum(D_inv)
            KSCMatrix = np.matmul(D_inv, np.matmul(MD, omega))

            eigv, alpha = np.linalg.eig(KSCMatrix)
            sorted_i = np.argsort(eigv)
            alpha = alpha[:][np._ix(sorted_i)]

            b = -(1 / np.sum(D_inv)) * np.sum(np.matmul(D_inv, np.matmul(omega, alpha)), 0)
            e = np.matmul(omega, alpha) + np.tile(b, [n_train, 1])

        for j in range(m):
            k = k_range(j)
            codebook = ksc_codebook(e[0:k - 1])
            q, s, cm = ksc_membership(e, codebook)

    return centres_init, alpha_init, q


def ksc_codebook(e):
    N, d = e.shape
    k = d + 1

    e_binary = np.sign(e)
    unique_cw, cw_indices, _, counts = np.unique(e_binary)

    sort_indices = np.argsort(counts)[::-1]

    codebook = unique_cw[sort_indices[0:k], :]

    # REMARK maybe cluster prototypes should be calculated here. That way points with deviant cluster indicators are not used for centre calculation
    # TODO calculate the centres of the alpha vectors per cluster indicator

    return codebook


def ksc_membership(e, codebook):
    """
        Calculates for each projected point (= rows in e) in which cluster it belongs. In case of soft clustering this is done in terms of probability
        :param e: Matrix where each row is the projection of a point
        :param codebook: Codebook as specified in KSC algorithm
        :param soft: If True the cluster membership (according to R. Lagone et al. SKSC)
        :return: 
        """
    k = codebook.shape[0]
    d = codebook.shape[1]
    n = e.shape[0]

    e_binary = np.sign(e)

    dist_matrix = spdist.cdist(e_binary, codebook, 'hamming')

    # vector with cluster id in each row.
    q = np.argmin(dist_matrix, axis=0)

    s = np.zeros([k, d])
    for i in range(k):
        s = np.mean(e[q == i], axis=1)

        e_norm = e/nplin.norm(e, axis=1)
        s_norm = s/nplin.norm(s, axis=1)
        d_cos = np.ones([n,k]) - np.matmul(e_norm, s_norm.transpose())

        #cluster membership
        prod_matrix = np.prod(d_cos,axis=1,keepdims=True)/d_cos
        cm = prod_matrix/np.sum(prod_matrix, axis=1, keepdims=True)

    return q, s, cm


def kernel_matrix(x, sigma, kernel_type='rbf'):
    # TODO: herschrijven in tensorflow indien te traag
    N = x.shape[0]
    d = x.shape[1]
    sigma_inv = np.linalg.inv(sigma)
    weighted_x = np.matmul(x, sigma)
    sq_x = np.sum(np.square(weighted_x), 1)

    omega = np.exp(-0.5 * (
    np.matmul(np.ones([N, 1]), np.transpose(sq_x)) - 2 * np.matmul(x, sigma_inv) + np.matmul(sq_x, np.ones([1, N]))))

    return omega
