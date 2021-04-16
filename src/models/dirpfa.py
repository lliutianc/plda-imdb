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


class DirPFAHMCSampler(PFAHMCSampler):
    '''
    When \theta and \phi both follow the Dirichlet distribution,
    and \H is an all-one matrix, PFA reduces to an equivalent form of the LDA.

    Note: There should be some modifications to the link function or the original data,
    since (\theta @ \phi) defines a *distribution* over vocabularies.
    '''
    param_names = ['l', 'theta', 'phi']

    def set_model(self, n):
        self.model = tfd.JointDistributionNamed(dict(

            l=tfd.Poisson(rate=self.hparam['n0']),

            theta=tfd.Dirichlet(self.hparam['topic_dist']),

            phi=tfd.Independent(tfd.Dirichlet(self.hparam['word_dist']), 1),

            document=lambda theta, phi, l: tfd.Independent(
                tfd.Poisson(rate=tf.matmul(theta, phi, adjoint_a=False) * l), ),

            ))

        self.unconstraining_bijectors = [tfb.Softplus(), tfb.SoftmaxCentered(),
                                         tfb.SoftmaxCentered()]

    def predict(self, full_generative_procese=False):
        if full_generative_procese:
            raise NotImplementedError()
        else:
            theta = self.states['theta']
            phi = self.states['phi']
            l = self.states['l']
            return tfd.Poisson(rate=tf.matmul(theta, phi) * l).sample()


if __name__ == '__main__':
    num_topic = 5
    vocab_size = 100
    n = 50
    document = generate_data(num_topic, vocab_size, n, partial_depend=True)

    hparams = {
        'n0': document.numpy().sum(1).mean(),
        'topic_dist': np.full((num_topic, ), 10.),
        'word_dist': np.full((num_topic, ), 1.)
        }

    model = DirPFAHMCSampler(vocab_size, num_topic, hparams)
    model.set_data(document)
    model.set_model(n)
    print(model.model.log_prob(document).shape)

    event = model.model.event_shape
    sample = model.model.sample()

    for k, v in model.model.batch_shape.items():
        print(f'{k}: {v}-{event[k]}')
        print(f'---{sample[k].shape}\n')