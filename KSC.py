import tensorflow as tf
import numpy as np
import scipy.spatial.distance as spdist
from sklearn.neighbors import NearestNeighbors
import numpy.linalg as nplin


class KSC:

    @staticmethod
    def ams(cm):
        """
        Average Membership Strength
        For each point the only the maximum cluster membership is kept, the rest is zeroed out. Then the sum is taken
        over all clusters
        :param cm:
        :return: int
        """
        n = cm.shape[0]
        k = cm.shape[1]

        indicator = np.zeros_like(cm)

        indicator = (cm == cm.max(axis=1)[:,np.newaxis]).astype(int)
        max_cm = cm*indicator

        count = np.sum(indicator, axis=0, keepdims=True)
        sum_cm = np.sum(max_cm, axis=0, keepdims=True)
        ams = (np.sum(sum_cm/count, axis=1)/k).flatten()
        return ams

    @staticmethod
    def ksc_codebook(e):
        N, d = e.shape
        k = d + 1

        e_binary = np.sign(e)
        unique_cw, cw_indices, counts = np.unique(e_binary, return_index=True, return_counts=True)

        if(len(unique_cw.shape)==1):
            unique_cw = np.expand_dims(unique_cw, -1)

        sort_indices = np.argsort(counts)[::-1]
        codebook = unique_cw[sort_indices[0:k], :]

        # REMARK maybe cluster prototypes should be calculated here.
        # That way points with deviant cluster indicators are not used for centre calculation
        # TODO calculate the centres of the alpha vectors per cluster indicator

        return codebook

    @staticmethod
    def cluster_membership(s, e):

        n = e.shape[0]
        k = s.shape[0]
        d = s.shape[1]
        e_norm = e / nplin.norm(e, axis=1, keepdims=True)
        s_norm = s / nplin.norm(s, axis=1, keepdims=True)
        d_cos = np.ones([n, k]) - np.matmul(e_norm, s_norm.transpose())

        # cluster membership:
        # cm is matrix with n rows and k columns
        prod_matrix = np.zeros([n, k])
        for i in range(k):
            d_cos_temp = d_cos.copy()
            d_cos_temp[:, i] = 1
            prod_matrix[:,i] = np.prod(d_cos_temp, axis=1)

        cm = prod_matrix / np.sum(prod_matrix, axis=1, keepdims=True)

        return cm

    @staticmethod
    def kernel_matrix(x, sigma, kernel_type='rbf'):
        # TODO: rewrite in tensorflow
        N = x.shape[0]
        d = x.shape[1]
        #sigma_inv = np.linalg.inv(sigma)
        #weighted_x = np.matmul(x, sigma)
        sq_x = np.sum(np.square(x), axis=1, keepdims=True)
        omega = np.exp(-0.5/np.square(sigma) * (
            np.matmul(np.ones([N, 1]), sq_x.T) - 2 * np.matmul(x, x.T) + np.matmul(sq_x,
                                                                                                     np.ones([1, N]))))
        return omega

    def __init__(self, x_train, k, sigma):

        # TODO create dimension convertor for sigma and k

        self.n = x_train.shape[0]
        self.d = x_train.shape[1]
        self.sigma = sigma
        self.k = k
        self.x_train = x_train

        self.q, self.e, self.alpha, _, self.cm, self.ams_train = self.construct()

    def construct(self, k=None , sigma=None):
        """
        Construct KSC model and return initializers for RBF Network
        """

        if k is None:
            k = self.k
        if sigma is None:
            sigma = self.sigma

        len_sigma = sigma.shape[0]
        len_k = k.shape[0]

        # calculate initial kernel
        for i in range(len_sigma):

            omega = self.kernel_matrix(self.x_train, sigma[i])
            omega = 0.5*(omega + omega.T)

            # degree matrix
            d_matrix = np.diag(np.sum(omega, 1))
            d_matrix_inv = np.linalg.inv(d_matrix)

            md_matrix = np.eye(self.n) - np.matmul(np.ones([self.n, self.n]), d_matrix_inv) / np.sum(d_matrix_inv)
            ksc_matrix = np.matmul(d_matrix_inv, np.matmul(md_matrix, omega))

            eigval, eigvec = np.linalg.eig(ksc_matrix)
            sorted_i = np.argsort(eigval)[::-1]

            for j in range(len_k):

                alpha = np.real(eigvec[np.ix_(sorted_i)][:, 0:k[i] - 1])
                b = -1 * (1 / np.sum(d_matrix_inv)) * np.sum(np.matmul(d_matrix_inv, np.matmul(omega, alpha)), axis=0,
                                                             keepdims=True)
                e = np.matmul(omega, alpha) + np.tile(b, [self.n, 1])

                self.codebook = KSC.ksc_codebook(e)
                q, s, cm = self._ksc_membership(e, self.codebook)
                ams = KSC.ams(cm)

        # TODO finish the implementation for k-ranges and sigma-ranges

        return q, e, alpha, s, cm, ams

    def create_initializers(self, k, sigma):

        pass

    def _ksc_membership(self, e, codebook):
        """
            Calculates for each projected point (= rows in e) in which cluster
            it belongs. In case of soft clustering this is done in terms of
            probability
            :param e: Matrix where each row is the projection of a point
            :param codebook: Codebook as specified in KSC algorithm
            :return:
            """
        k = codebook.shape[0]
        d = codebook.shape[1]
        n = e.shape[0]

        e_binary = np.sign(e)

        dist_matrix = spdist.cdist(e_binary, codebook, 'hamming')

        # vector with cluster id in each row.
        q = np.argmin(dist_matrix, axis=1)
        # TODO kNN implementatie
        #knn = NearestNeighbors(3,metric="hamming")
        #knn.fit(codebook)
        #_, q = knn.kneighbors(e_binary)

        s = np.zeros([k, d])

        # s is a prototype for each cluster in the e-space
        for i in range(k):

            s[i, :] = np.mean(e[q == [i]], axis=0)

        cm = KSC.cluster_membership(s, e)
        return q, s, cm
