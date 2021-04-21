import warnings

from src.math import to_float, finite_reduce_sum
from src.data import generate_data

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

hmc = tfp.mcmc.HamiltonianMonteCarlo
stepAdapt = tfp.mcmc.SimpleStepSizeAdaptation
sample_chain = tfp.mcmc.sample_chain


class PFAHMCSampler:
    param_names = ['gamma0', 'gamma', 'theta', 'phi']

    def __init__(self, vocab_size, num_topic, hparam, seed=1, ):
        self.num_topic = num_topic
        self.vocab_size = vocab_size
        self.set_hparam(hparam)
        self.seed = seed
        warnings.warn('Didn\'t set random seed in fact.')
        self.init__model()

    def init__model(self):
        # Training states
        self.states = None
        self.n_states = None
        self.model = None
        # Converge states
        self._rhat = None
        self._essr = None
        self.document = None

    def set_data(self, document):
        assert document.shape[1] == self.vocab_size
        self.document = document

    def store_states(self, states):
        assert len(states) == len(self.__class__.param_names)
        self.states = { }
        for idx, param_name in enumerate(self.__class__.param_names):
            self.states[param_name] = states[idx]

    def set_hparam(self, hparam):
        self.hparam = { }
        for k, v in hparam.items():
            v = v.astype(np.float32) if isinstance(v, np.ndarray) else float(v)
            self.hparam[k] = tf.Variable(v, name=k)

    def init_states(self):
        assert self.document is not None
        assert self.model is not None

        state = self.model.sample()
        return [state[param] for param in self.__class__.param_names]

        # return [state[param] * 0. + 1. for param in self.__class__.param_names]

    def log_prob(self, states, reduce=True):
        states_dict = { 'document': self.document }
        for k, v in zip(self.__class__.param_names, states):
            states_dict[k] = v

        if reduce:
            return tf.reduce_sum(self.model.log_prob(states_dict))
        else:
            return self.model.log_prob(states_dict)

    def sample_states(self, document=None, n_states=100, n_thin=0, n_burnin=100, step_size=1e-2,
                      num_step=1, num_leapfrog_steps=3, num_adaptation_steps=None, ):
        if document is not None:
            self.document = document
        assert self.document is not None, 'Training document must be provided first.'

        n, v = self.document.shape
        assert v == self.vocab_size
        self.n_states = n_states
        self.set_model(n)

        def _log_prob(*params):
            return self.log_prob(params)

        trace_fn = lambda _, pkr: pkr.inner_results.inner_results.is_accepted
        num_adaptation_steps = num_adaptation_steps or self.n_states

        adaptive_hmc = stepAdapt(
            tfp.mcmc.TransformedTransitionKernel(
            hmc(_log_prob, step_size, num_leapfrog_steps),
            bijector=self.unconstraining_bijectors),
            num_adaptation_steps)

        states, is_accepted = sample_chain(
            kernel=adaptive_hmc,
            num_results=n_states,
            num_burnin_steps=n_burnin,
            num_steps_between_results=n_thin,
            trace_fn=trace_fn,
            current_state=self.init_states())

        self.store_states(states)
        self.is_accepted = is_accepted
        print('Finish sampling...')

    def predict(self, full_generative_procese=False):
        if full_generative_procese:
            raise NotImplementedError()
        else:
            return tfd.Poisson(rate=tf.matmul(
                self.states['theta'], self.states['phi'], adjoint_a=False)).sample()

    def set_model(self, n):
        self.model = tfd.JointDistributionNamed(dict(

            # phi=tfd.Independent(tfd.Dirichlet(
            #     tfp.util.TransformedVariable(self.hparam['word_dist'], tfb.Softplus())), 1),

            gamma0=tfd.Gamma(concentration=self.hparam['e0'], rate=self.hparam['f0']),

            gamma=lambda gamma0: tfd.Independent(tfd.Gamma(
                concentration=gamma0, rate=self.hparam['c0']), 1),

            theta=lambda gamma: tfd.Independent(tfd.Gamma(
                concentration=gamma, rate=self.hparam['pn'] / (1 - self.hparam['pn'])),1),

            phi=tfd.Independent(tfd.Dirichlet(self.hparam['word_dist']), 1),

            document=lambda theta, phi: tfd.Independent(
                tfd.Poisson(rate=tf.matmul(theta, phi, adjoint_a=False)), )
            ))

        self.unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(), tfb.Softplus(), tfb.SoftmaxCentered(), ]



if __name__ == '__main__':
    num_topic = 5
    vocab_size = 100
    n = 50
    document = generate_data(num_topic, vocab_size, n, partial_depend=True)

    hparams = { 'e0': 1.,
                'f0': 0.001,
                'c0': np.full((num_topic,), 0.5),
                'pn': np.full((n, 1), 0.5),
                'word_dist': np.full((num_topic, vocab_size), 10.),
        }

    model = PFAHMCSampler(vocab_size, num_topic, hparams)

    # n_states = 100
    # n_burnin = 500
    # model.sample_states(document,
    #                   n_states=n_states,
    #                   n_burnin=n_burnin,
    #                   step_size=0.005)

    model.set_data(document)
    model.set_model(n)
    model.model.log_prob(model.model.sample())
    event = model.model.event_shape

    for k, v in model.model.batch_shape.items():
        print(f'{k}: {v}-{event[k]} \n')

    # draw_samples = pfa.predict()
    # pred_low, pred_high = np.quantile(draw_samples, [0.025, 0.975], axis=0)
    # is_cover = np.all((pred_low < document, pred_high > document), axis=0)

    # print(is_cover.mean(0))