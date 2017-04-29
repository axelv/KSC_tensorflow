import tensorflow as tf
import io
import numpy as np
from sklearn.decomposition import PCA,KernelPCA
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot2D( x, y='b', title="Normal Scatter Plot", xlabel="1st dimension", ylabel="2nd dimension", axes=None):

    n = x.shape[0]

    if x.shape[1] == 1:
        x = np.concatenate([x, np.zeros_like(x)], axis=1)

    fig = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if axes is not None:
        plt.axis(axes)
    plt.title(title)


    return fig



def plot3D(x):
    pass


def plot_3pcs(x, y=None, title="Principal Components", xlabel="1st dimension", ylabel="2nd dimension",
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

def plot_2kpcs(x, sigma, y=None, title="Kernel Principal Components", xlabel="1st dimension", ylabel="2nd dimension"):

    n = x.shape[0]
    d = x.shape[1]

    if y is None:
        y = np.zeros([n, 1])

    gamma = 0.5/np.square(sigma)
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=gamma)
    x_red = kpca.fit_transform(x)

    fig = plt.figure()
    plt.scatter(x_red[:, 0], x_red[:, 1], c=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return fig


def process_plot(fig):

    fig.canvas.draw()
    l, b, w, h = fig.bbox.bounds
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    data.shape = h.round().astype('int32'), w.round().astype('int32'), 4
    data = np.expand_dims(data, 0);
    return data


def show():
    plt.show()

