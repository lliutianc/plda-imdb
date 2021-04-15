from src.math import to_float, finite_reduce_sum

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

        # Training states
        self.states = None
        self.n_states = None
        self.model = None
        # Converge states
        self._rhat = None
        self._essr = None
        self.document = None

        self.num_topic = num_topic
        self.vocab_size = vocab_size
        self.set_hparam(hparam)
        self.seed = seed

    def set_data(self, document):
        assert document.shape[1] == self.vocab_size
        self.document = document

    def store_states(self, states):
        assert len(states) == len(PFAHMCSampler.param_names)
        self.states = { }
        for idx, param_name in enumerate(PFAHMCSampler.param_names):
            self.states[param_name] = states[idx]

    def set_hparam(self, hparam):
        self.hparam = { }

        self.hparam['c0'] = tf.Variable(hparam['c0'], name='c0')
        self.hparam['e0'] = tf.Variable(hparam['e0'], name='e0')
        self.hparam['f0'] = tf.Variable(hparam['f0'], name='f0')
        self.hparam['pn'] = tf.Variable(hparam.get('pn', .5), name='pn')

        if 'alpha_vec' not in hparam:
            hparam['alpha_vec'] = np.full((self.vocab_size,), 10.)
        else:
            assert hparam['alpha_vec'].shape == (self.vocab_size,)

        hparam['alpha_vec'] = hparam['alpha_vec'].astype(np.float32)
        self.hparam['alpha_vec'] = tf.Variable(hparam['alpha_vec'],
                                               name='alpha_vec')

    def init_states(self):
        assert self.document is not None
        assert self.model is not None

        state = self.model.sample()
        state = [state[param] for param in PFAHMCSampler.param_names]

        return state

    def set_model(self, n):
        self.model = tfd.JointDistributionNamed(dict(
            phi=tfd.Sample(tfd.Dirichlet(
                tfp.util.TransformedVariable(self.hparam['alpha_vec'], tfb.Softplus())),
                sample_shape=(self.num_topic,)),

            gamma0=tfd.Gamma(concentration=self.hparam['e0'],
                             rate=self.hparam['f0']),

            gamma=lambda gamma0: tfd.Sample(
                tfd.Gamma(concentration=gamma0, rate=self.hparam['c0']),
                sample_shape=(self.num_topic,)),

            theta=lambda gamma: tfd.Independent(tfd.Sample(
                tfd.Gamma(concentration=gamma,
                          rate=self.hparam['pn'] / (1 - self.hparam['pn']), ),
                sample_shape=(n,)), 1),

            document=lambda theta, phi: tfd.Independent(
                tfd.Poisson(rate=tf.matmul(theta, phi, adjoint_a=True)), 1)))

    def log_prob(self, states):
        states_dict = {'document': self.document}
        for k, v in zip(PFAHMCSampler.param_names, states):
            states_dict[k] = v

        log_prob = self.model.log_prob(states_dict)
        if tf.math.is_nan(log_prob).numpy().max() is True:
            print(log_prob)
            self.final_state = states
            raise ValueError

        return tf.reduce_sum(log_prob)

    def sample_states(self,
                      document=None,
                      n_states=100,
                      n_thin=0,
                      n_burnin=100,
                      step_size=1e-2,
                      num_leapfrog_steps=3,
                      num_adaptation_steps=None,
                      ):
        if document is not None:
            self.document = document
        assert self.document is not None, \
            'Training document must be provided first.'

        n, v = self.document.shape
        assert v == self.vocab_size
        self.n_states = n_states
        self.set_model(n)

        def _log_prob(*params):
            return self.log_prob(params)

        trace_fn = lambda _, pkr: pkr.inner_results.is_accepted
        num_adaptation_steps = num_adaptation_steps or self.n_states

        hmc_core = hmc(_log_prob, step_size, num_leapfrog_steps)
        adaptive_hmc = stepAdapt(hmc_core, num_adaptation_steps)
        states, is_accepted = sample_chain(num_results=n_states,
                                           num_burnin_steps=n_burnin,
                                           num_steps_between_results=n_thin,
                                           kernel=adaptive_hmc,
                                           trace_fn=trace_fn,
                                           current_state=self.init_states())
        self.store_states(states)
        self.is_accepted = is_accepted

    def predict(self, full_generative_procese=False):
        if full_generative_procese:
            raise NotImplementedError()
        else:
            return tfd.Poisson(
                rate=tf.matmul(self.states['theta'], self.states['phi'], adjoint_a=True)).sample()



def generate_data():
    e0 = 1
    f0 = 0.001
    c0 = 1.
    pn = 0.8
    n_chain = 1

    alpha_vec = tf.fill((vocab_size,), value=1.01)
    alpha_vec = np.cumsum(alpha_vec) / 5.
    # alpha_vec = alpha_vec / alpha_vec.sum()
    succ = False
    for _ in range(1000):
        phi = tfd.Dirichlet(tfp.util.TransformedVariable(alpha_vec, tfb.Softplus())).sample(
            (n_chain, num_topic))
        # phi = tf.transpose(phi, [0, 1, 2])
        assert phi.shape[1:] == (num_topic, vocab_size)

        gamma0 = tfd.Gamma(concentration=e0, rate=f0).sample((n_chain,))
        assert gamma0.shape[1:] == ()

        gamma = tfd.Gamma(concentration=gamma0, rate=c0).sample((num_topic,))
        gamma = tf.transpose(gamma, [1, 0])
        assert gamma.shape[1:] == (num_topic,)

        theta = tfd.Gamma(concentration=gamma, rate=(1 - pn) / pn).sample((n,))
        theta = tf.transpose(theta, [1, 2, 0])
        assert theta.shape[1:] == (num_topic, n)

        doc_mean = tf.matmul(phi, theta, adjoint_a=True)
        if doc_mean.numpy().min() > 3:
            succ = True
            break

    if not succ:
        raise ValueError('Unsuccess model')

    document = tf.random.poisson(lam=doc_mean, shape=())[0]
    document = tf.transpose(document, [1, 0])
    assert document.shape == (n, vocab_size)

    return document


if __name__ == '__main__':
    num_topic = 5
    vocab_size = 100
    n = 50

    document = generate_data()

    hparams = {'c0': 1.,
               'e0': 1.,
               'f0': 0.001,
               }
    pfa = PFAHMCSampler(vocab_size, num_topic, hparams)

    n_states = 100
    n_burnin = 100
    pfa.sample_states(document,
                      n_states=n_states,
                      n_burnin=n_burnin,
                      step_size=0.005)

    draw_samples = pfa.predict()
    pred_low, pred_high = np.quantile(draw_samples, [0.025, 0.975], axis=0)
    is_cover = np.all((pred_low < document, pred_high > document), axis=0)

    print(is_cover.mean(0))