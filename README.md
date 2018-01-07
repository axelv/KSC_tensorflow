# KSC_tensorflow

Kernel Spectral Clustering (KSC) is a clustering technique combining ideas from Spectral Clustering and Weighted Kernel PCA.
A full description of the algorithm is available in [Multiway Spectral Clustering with Out-of-Sample Extension through Weighted Kernel PCA - Alzate et al.](https://www.google.be/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwjDzcWI8r3YAhURbFAKHUApBmIQFggtMAE&url=https%3A%2F%2Fwww.esat.kuleuven.be%2Fsista%2Flssvmlab%2Fpami2010.pdf&usg=AOvVaw0xUsZwLYWdXsANw_lUdQMO).
The paper also contains a MATLAB implementation. This project is a Scipy/TensorFlow implementation of the algorithm plus a online extension.

## Online extension
As noted in the paper, the clustering algorithm can be framed as a [Least Squares Support Vector Machine - Suykens et al.](https://www.google.be/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjJufr08r3YAhUQL1AKHf2QAtAQFggnMAA&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FLeast_squares_support_vector_machine&usg=AOvVaw2-kgtZTvITXp5ztjvVW38x) which can be viewed as a special solution of RBF-networks.
The idea in this project was to use the KSC algorithm as a smart initialization of the RBF-network. This way it is possible to put Spectral Clustering in a neural network form and combine it with other neural network layers. Subsequently backpropagation is used to update the clusters centers and the weight vectors.
Tests are done with slowly moving clusters. However, the special structure of the KSC-eigenvectors is completetly lost and there is no clear way to add/remove clusters.

<b>TODO</b>: A possible improvement is to use [Kernel Hebbian Algorithm for Iterative KPCA - Kim, Franz, Scholkopf](http://ieeexplore.ieee.org/abstract/document/1471703/)
to update the weights of the network.

## File Description

- [KSC.py](./KSC.py): the full KSC algorithm. The terminology used in the comments is conform the paper. An example on how to use the algorithm standalone is available in [KSC_test.py](./KSC_test.py)
- [layers.py](./layers.py): the necessary neural network layers written in TensorFlow which make it possible to frame the KSC-algorithm in a RBF-network.
- [network_test.py](./network_test.py): a test of the out-of-sample extension of the clustering algorithm using the layers from [layers.py](./layers.py)
- [backprop_test.py](./backprop_test.py): a test using backpropagation on the above-mentioned RBF-network with slowly moving clusters. The cluster results are plotted in the image tab in TensorBoard.
