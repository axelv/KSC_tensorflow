import KSC
import numpy as np
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=400, factor=0.3, noise=.05)

sigma =  np.expand_dims(np.eye(2)*0.22,0)
k = [2]

ksc_model = KSC.KSC(X, k, sigma)
