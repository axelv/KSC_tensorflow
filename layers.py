import tensorflow as tf

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
