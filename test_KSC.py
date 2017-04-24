import KSC
import numpy as np
import plot_tools as plt
import io_tools as iot
from sklearn.datasets import make_circles

#X, y = make_circles(n_samples=400, factor=0.3, noise=.05)
data = iot.from_matlab("numpy")
#iot.to_matlab([X, y])
X = data['numpy0']
y = data['numpy1']

#sigma =  np.expand_dims(np.eye(2)*0.001,0)
sigma_f = 0.1

sigma = np.array([sigma_f])
k = np.array([2])

ksc_model = KSC.KSC(X, k, sigma)

print("AMS: "+str(ksc_model.ams_train))
print("Done")

plt.plot2D(X,y, title="Ground truth")
plt.plot2D(X, ksc_model.cm[:,1], title="KSC")
plt.plot_2kpcs(X,sigma_f, ksc_model.cm[:,1], title="Kernel PCA")
plt.plot2D(ksc_model.e, y, title="KSC - score variables")
plt.plot2D(ksc_model.alpha, ksc_model.cm[:,1], title="KSC - eigenvectors")
plt.show()

