import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def plot_pcs(x, y=None, title="Principal Components", xlabel="1st dimension", ylabel="2nd dimension",
             zlabel="3rd dimension"):

    n = x.shape[0]
    d = x.shape[1]

    if y is None:
        y = np.zeros([n, 1])

    x_3d = PCA(n_components=3).fit_transform(x)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_3d[:, 0], x_3d[:, 1], x_3d[:, 2], c=y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    return fig
