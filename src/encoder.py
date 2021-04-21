import theano.tensor as tt
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStreams

import numpy as np

class ThetaEncoder:
    """Encode (term-frequency) document vectors to variational means and (log-transformed) stds."""

    def __init__(self, n_words, n_hidden, n_topics, random_seed=1):
        rng = np.random.RandomState(random_seed)
        self.n_words = n_words
        self.n_hidden = n_hidden
        self.n_topics = n_topics
        self.w0 = shared(0.01 * rng.randn(n_words, n_hidden).ravel(), name="w0")
        self.b0 = shared(0.01 * rng.randn(n_hidden), name="b0")

        self.w1 = shared(0.01 * rng.randn(n_hidden, n_hidden).ravel(), name="w1")
        self.b1 = shared(0.01 * rng.randn(n_hidden), name="b1")

        self.w2 = shared(0.01 * rng.randn(n_hidden, 2 * (n_topics - 1)).ravel(), name="w2")
        self.b2 = shared(0.01 * rng.randn(2 * (n_topics - 1)), name="b2")
        self.rng = MRG_RandomStreams(seed=random_seed)

    def encode(self, xs):
        w0 = self.w0.reshape((self.n_words, self.n_hidden))
        w1 = self.w1.reshape((self.n_hidden, self.n_hidden))
        w2 = self.w2.reshape((self.n_hidden, 2 * (self.n_topics - 1)))

        hs = tt.tanh(xs.dot(w0) + self.b0)
        hs = tt.tanh(hs.dot(w1) + self.b1)
        zs = hs.dot(w2) + self.b2
        zs_mean = zs[:, : (self.n_topics - 1)]
        zs_rho = zs[:, (self.n_topics - 1):]
        return { "mu": zs_mean, "rho": zs_rho }

    def get_params(self):
        return [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2]


class ThetaNEncoder:
    """Encode (term-frequency) document vectors to variational means and (log-transformed) stds."""

    def __init__(self, n_words, n_hidden, n_topics, random_seed=1):
        rng = np.random.RandomState(random_seed)
        self.n_words = n_words
        self.n_hidden = n_hidden
        self.n_topics = n_topics
        self.w0 = shared(0.01 * rng.randn(n_words, n_hidden).ravel(), name="w0")
        self.b0 = shared(0.01 * rng.randn(n_hidden), name="b0")

        self.w1 = shared(0.01 * rng.randn(n_hidden, n_hidden).ravel(), name="w1")
        self.b1 = shared(0.01 * rng.randn(n_hidden), name="b1")

        self.w2 = shared(0.01 * rng.randn(n_hidden, 2 * (n_topics - 1)).ravel(), name="w2")
        self.b2 = shared(0.01 * rng.randn(2 * (n_topics - 1)), name="b2")

        self.w3 = shared(0.01 * rng.randn(n_hidden, 2).ravel(), name="w3")
        self.b3 = shared(0.01 * rng.randn(2), name="b3")

        self.rng = MRG_RandomStreams(seed=random_seed)

    def encode(self, xs):
        w0 = self.w0.reshape((self.n_words, self.n_hidden))
        w1 = self.w1.reshape((self.n_hidden, self.n_hidden))
        w2 = self.w2.reshape((self.n_hidden, 2 * (self.n_topics - 1)))
        w3 = self.w3.reshape((self.n_hidden, 2))

        hs = tt.tanh(xs.dot(w0) + self.b0)
        hs = tt.tanh(hs.dot(w1) + self.b1)
        thetas = hs.dot(w2) + self.b2
        rates = hs.dot(w3) + self.b3

        thetas_mean = thetas[:, :(self.n_topics - 1)]
        thetas_rho = thetas[:, (self.n_topics - 1):]

        rates_mean = rates[:, :1]
        rates_rho = rates[:, 1:]
        return { "mu": thetas_mean, "rho": thetas_rho }, { "mu": rates_mean, "rho": rates_rho }

    def get_params(self):
        return [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]