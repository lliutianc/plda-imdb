from src.math import to_float, finite_reduce_sum
from src.models.pfa import PFAHMCSampler
from src.data import generate_data

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

hmc = tfp.mcmc.HamiltonianMonteCarlo
stepAdapt = tfp.mcmc.SimpleStepSizeAdaptation
sample_chain = tfp.mcmc.sample_chain


class DPFAHMCSampler(PFAHMCSampler):
    """
    Deep PFA (SBN) introduces a two-layer prior on the H matrix, namely,
    P(H_0) \in \{ 0, 1 \}^{K_2}
    W \sim Gaussian, (we place N(0, 10I) here)
    p(H_k = 1 \mid H_0) = \sigmoid ( c_k + W_k @ H_0 ), W \in \real^{K \times K_2}, c
    """
    param_names = ['w1', 'h0', 'h', 'gamma0', 'gamma', 'theta', 'phi']

    def set_model(self, n):
        self.model = tfd.JointDistributionNamed(dict(

            # h0=tfd.Independent(tfd.Beta(concentration1=tf.ones([n, self.num_topic]),
            #                             concentration0=tf.ones([n, self.num_topic]),
            #                             )),

            w1=tfd.Independent(tfd.Normal(loc=0., scale=self.hparam['w1_std'])),

            h0=tfd.Independent(tfd.Bernoulli(logits=self.hparam['bias0'], dtype=tf.float32), 2),

            h=lambda h0, w1: tfd.Independent(
                tfd.Bernoulli(dtype=tf.float32, logits=self.hparam['bias1'] + tf.matmul(w1, h0))),

            gamma0=tfd.Gamma(concentration=self.hparam['e0'], rate=self.hparam['f0']),

            gamma=lambda gamma0: tfd.Independent(tfd.Gamma(
                concentration=gamma0, rate=self.hparam['c0']), 1),

            theta=lambda gamma: tfd.Independent(
                tfd.Gamma(concentration=gamma, rate=self.hparam['pn'] / (1 - self.hparam['pn'])),
                1),

            phi=tfd.Independent(tfd.Dirichlet(self.hparam['word_dist']), 1),

            document=lambda theta, phi, h: tfd.Independent(
                tfd.Poisson(rate=tf.matmul(theta * h, phi, adjoint_a=False)), )

            ))

        self.unconstraining_bijectors = [
            tfb.Identity(),
            tfb.Sigmoid(),
            tfb.Sigmoid(),
            tfb.Softplus(),
            tfb.Softplus(),
            tfb.Softplus(),
            tfb.SoftmaxCentered(),]

    def predict(self, full_generative_procese=False):
        if full_generative_procese:
            raise NotImplementedError()
        else:
            return tfd.Poisson(rate=tf.matmul(
                self.states['theta'] * self.states['h'],
                self.states['phi'], adjoint_a=False)).sample()


if __name__ == '__main__':
    num_topic = 5
    vocab_size = 100
    n = 50
    document = generate_data(num_topic, vocab_size, n, partial_depend=True)

    hparams = {
                'e0': 1.,
                'f0': 0.001,
                'c0': np.full((num_topic,), 0.5),
                'pn': np.full((n, 1), 0.5),
                'word_dist': np.full((num_topic, vocab_size), 10.),
                'bias0': np.random.normal(size=(num_topic, num_topic)) * .1,
                'bias1': np.random.normal(size=(1, num_topic)) * .1,
                'w1_std': np.full((n, num_topic), 10.),
                }

    model = DPFAHMCSampler(vocab_size, num_topic, hparams)
    model.set_data(document)
    model.set_model(n)
    print(model.model.log_prob(model.model.sample()))

    event = model.model.event_shape
    sample = model.model.sample()

    for k, v in model.model.batch_shape.items():
        print(f'{k}: {v}-{event[k]}')
        print(f'---{sample[k].shape}\n')

    # n_states = 100
    # n_burnin = 500
    # model.sample_states(document,
    #                   n_states=n_states,
    #                   n_burnin=n_burnin,
    #                   step_size=0.005)
    #
    # draw_samples = model.predict()
    # pred_low, pred_high = np.quantile(draw_samples, [0.025, 0.975], axis=0)
    # is_cover = np.all((pred_low < document, pred_high > document), axis=0)
    #
    # print(is_cover.mean(0))
    # print('-' * 100)
    # print(pred_low)
    # print('-' * 100)
    # print(pred_high)
    # print('-' * 100)
    # print(document)