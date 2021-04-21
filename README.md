# plda-imdb

This project implements Latent Dirichlet Allocation (LDA), a shallow Poisson Factor Analysis (PFA) without binary mask matrix H, and a new Poisson-LDA model. 

The implmentation uses the autoencoding variational Bayes to learn the approximation of topic distribution/intensity for each document and is updated with mini-batch data.

The implementation is based on PyMC3 and Theano libaries.

HMC based implementations on word count matrices are also included but may be buggy. They are stored under the `src/old` folders and are based on `tensorflow_probability`.
