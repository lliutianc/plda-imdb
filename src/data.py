import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


def generate_data(num_topic, vocab_size, n, partial_depend=True):
    np.random.seed(1)

    e0 = 1
    f0 = 0.001
    c0 = 1.
    pn = 0.8
    n_chain = 1

    alpha_vec = tf.fill((vocab_size,), value=1.01)
    alpha_vec = np.cumsum(alpha_vec) / 5.

    if partial_depend:
        p = np.r_[[np.array([.4, .5, .6, .7, .8]) for _ in range(n)]].T
        tfd = tfp.distributions
        while 1:
            h = tfd.Binomial(total_count=1., probs=p.astype(np.float32)).sample()
            if h.numpy().sum(0).min() > 0:
                break
    else:
        h = 1.

    succ = False
    for _ in range(1000):
        phi = tfd.Dirichlet(tfp.util.TransformedVariable(alpha_vec, tfb.Softplus())).sample(
            (n_chain, num_topic))
        assert phi.shape[1:] == (num_topic, vocab_size)


        gamma0 = tfd.Gamma(concentration=e0, rate=f0).sample((n_chain,))
        assert gamma0.shape[1:] == ()

        gamma = tfd.Gamma(concentration=gamma0, rate=c0).sample((num_topic,))
        gamma = tf.transpose(gamma, [1, 0])
        assert gamma.shape[1:] == (num_topic,)

        theta = tfd.Gamma(concentration=gamma, rate=(1 - pn) / pn).sample((n,))
        theta = tf.transpose(theta, [1, 2, 0])
        assert theta.shape[1:] == (num_topic, n)

        doc_mean = tf.matmul(phi, theta * h, adjoint_a=True)
        if np.quantile(doc_mean.numpy(), .05) > 3:
            succ = True
            break

    if not succ:
        raise ValueError('Unsuccess model')

    document = tf.random.poisson(lam=doc_mean, shape=())[0]
    document = tf.transpose(document, [1, 0])
    assert document.shape == (n, vocab_size)

    return document

