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

        centres = tf.get_variable("centres", [n_centres, d_input], initializer=centres_init, dtype=tf.float64)
        sigma = tf.get_variable("sigma", [d_input, d_input], initializer=sigma_init, dtype=tf.float64)

        # operation body
        x_ext = tf.tile(tf.expand_dims(x, 1), [1, n_centres, 1])

        # d² = (x-c)^T*SIGA^-1*(x-c)
        sigma_inv = tf.matrix_inverse(sigma)
        diff = x_ext - centres  # [batch_size, n_centres, d_input]
        diff_shaped = tf.reshape(diff, [-1, d_input])  # [batch_size*n_centres, d_input]
        weighted_diff = tf.matmul(diff_shaped, sigma_inv)  # [batch_size*n_centres, d_input]
        weighted_diff_shaped = tf.reshape(weighted_diff,[-1, n_centres, d_input])

        sqrd_dist_c = tf.reduce_sum(tf.square(weighted_diff_shaped), -1)
        y = tf.exp(-0.5 * sqrd_dist_c)  # [batch_size, n_centres]

    return y


def linear_layer(x, size, weights_init=tf.random_normal_initializer(), biases_init=tf.random_normal_initializer(),
                 name="Linear_layer", reuse=None):
    """
    Linear Neural Network Layer
    """
    with tf.variable_scope(name, reuse=reuse):

        weights = tf.get_variable("alpha", size, initializer=weights_init, dtype=tf.float64)
        biases = tf.get_variable("biases", [1, size[1]], initializer=biases_init, dtype=tf.float64)

        y = tf.matmul(x, weights) + biases  # [batch_size, l]

    return y


def cosine_layer(x, k, weights_init=tf.random_normal_initializer(), name="Cosine_layer", reuse=None):
    """
    Cosine Distance Layer
    """
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable("prototypes", [k-1, k], initializer= weights_init, dtype=tf.float64)
        y = tf.matmul(x, weights)
        y = y / tf.norm(y)

    return y
