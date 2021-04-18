import os, sys, argparse
import pickle
from collections import OrderedDict
from copy import deepcopy
from time import time

import numpy as np
import scipy
import scipy.special as sc
import seaborn as sns
import matplotlib.pyplot as plt

import pymc3 as pm
import theano
import theano.tensor as tt

from pymc3 import Dirichlet, Poisson, Gamma
from pymc3 import math as pmmath
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStreams

from data_prep import prepare_sparse_matrix
from encoder import ThetaEncoder, ThetaNEncoder
from utils import Logger, makedirs


def prepare_sparse_matrix_nonlabel(n_train, n_test, max_vocab_size):
    return prepare_sparse_matrix(n_train, n_test, max_vocab_size)[:3]


def run_lda(args):
    tf_vectorizer, docs_tr, docs_te = prepare_sparse_matrix_nonlabel(args.n_tr, args.n_te, args.n_word)
    feature_names = tf_vectorizer.get_feature_names()
    doc_tr_minibatch = pm.Minibatch(docs_tr.toarray(), args.bsz)
    doc_tr = shared(docs_tr.toarray()[:args.bsz])

    def log_prob(beta, theta):
        """Returns the log-likelihood function for given documents.

        K : number of topics in the model
        V : number of words (size of vocabulary)
        D : number of documents (in a mini-batch)

        Parameters
        ----------
        beta : tensor (K x V)
            Word distributions.
        theta : tensor (D x K)
            Topic distributions for documents.
        """

        def ll_docs_f(docs):
            dixs, vixs = docs.nonzero()
            vfreqs = docs[dixs, vixs]
            ll_docs = (vfreqs * pmmath.logsumexp(tt.log(theta[dixs]) + tt.log(beta.T[vixs]),
                                                 axis=1).ravel())

            return tt.sum(ll_docs) / (tt.sum(vfreqs) + 1e-9)

        return ll_docs_f

    with pm.Model() as model:
        theta = Dirichlet("theta",
                          a=pm.floatX((1.0 / args.n_topic) * np.ones((args.bsz, args.n_topic))),
                          shape=(args.bsz, args.n_topic), total_size=args.n_tr, )

        beta = Dirichlet("beta",
                         a=pm.floatX((1.0 / args.n_topic) * np.ones((args.n_topic, args.n_word))),
                         shape=(args.n_topic, args.n_word), )

        doc = pm.DensityDist("doc", log_prob(beta, theta), observed=doc_tr)

    encoder = ThetaEncoder(n_words=args.n_word, n_hidden=100, n_topics=args.n_topic)
    local_RVs = OrderedDict([(theta, encoder.encode(doc_tr))])
    encoder_params = encoder.get_params()

    s = shared(args.lr)

    def reduce_rate(a, h, i):
        s.set_value(args.lr / ((i / args.bsz) + 1) ** 0.7)

    with model:
        approx = pm.MeanField(local_rv=local_RVs)
        approx.scale_cost_to_minibatch = False
        inference = pm.KLqp(approx)

    inference.fit(args.n_iter,
                  callbacks=[reduce_rate, pm.callbacks.CheckParametersConvergence(diff="absolute")],
                  obj_optimizer=pm.adam(learning_rate=s),
                  more_obj_params=encoder_params,
                  total_grad_norm_constraint=200,
                  more_replacements={ doc_tr: doc_tr_minibatch }, )

    doc_tr.set_value(docs_tr.toarray())
    inp = tt.matrix(dtype="int64")
    sample_vi_theta = theano.function([inp],
        approx.sample_node(approx.model.theta, args.n_sample, more_replacements={doc_tr: inp}), )

    test = docs_te.toarray()
    test_n = test.sum(1)

    beta_pymc3 = pm.sample_approx(approx, draws=args.n_sample)['beta']
    theta_pymc3 = sample_vi_theta(test)

    assert beta_pymc3.shape == (args.n_sample, args.n_topic, args.n_word)
    assert theta_pymc3.shape == (args.n_sample, args.n_te, args.n_topic)

    beta_mean = beta_pymc3.mean(0)
    theta_mean = theta_pymc3.mean(0)

    pred_rate = theta_mean.dot(beta_mean)
    pp_test = (test * np.log(pred_rate)).sum(1) / test_n

    posteriors = { 'theta': theta_pymc3, 'beta': beta_pymc3,}

    log_top_words(beta_pymc3.mean(0), feature_names, n_top_words=args.n_top_word)
    save_elbo(approx.hist)
    save_pp(pp_test)
    save_draws(posteriors)


def run_pfa(args):
    tf_vectorizer, docs_tr, docs_te = prepare_sparse_matrix_nonlabel(args.n_tr, args.n_te, args.n_word)
    feature_names = tf_vectorizer.get_feature_names()
    doc_tr_minibatch = pm.Minibatch(docs_tr.toarray(), args.bsz)
    doc_tr = shared(docs_tr.toarray()[:args.bsz])

    def log_prob(beta, theta):
        """Returns the per-word log-likelihood function for given documents.

        K : number of topics in the model
        V : number of words (size of vocabulary)
        D : number of documents (in a mini-batch)

        Parameters
        ----------
        beta : tensor (K x V)
            Word distributions.
        theta : tensor (D x K)
            Topic distributions for documents.
        """

        def ll_docs_f(docs):
            dixs, vixs = docs.nonzero()
            vfreqs = docs[dixs, vixs]
            ll_docs = ((theta[dixs] + beta.T[vixs]).sum(1) + vfreqs * pmmath.logsumexp(
                tt.log(theta[dixs]) + tt.log(beta.T[vixs]),
                axis=1).ravel() - pm.distributions.special.gammaln(vfreqs + 1))

            ll_docs = (vfreqs * (pmmath.logsumexp(tt.log(theta[dixs]) + tt.log(beta.T[vixs]),
                                                  axis=1).ravel()) - tt.exp(
                pmmath.logsumexp(tt.log(theta[dixs]) + tt.log(beta.T[vixs], ),
                                 axis=1)).ravel() - pm.distributions.special.gammaln(vfreqs + 1))

            return tt.sum(ll_docs) / (tt.sum(vfreqs) + 1e-9)

        return ll_docs_f

    e0 = c0 = 1.
    f0 = .01
    pn = .5

    with pm.Model() as model:
        beta = Dirichlet("beta", a=pm.floatX((1.0 / args.n_topic) * np.ones((args.n_topic, args.n_word))),
            shape=(args.n_topic, args.n_word), )

        gamma0 = Gamma("gamma0", alpha=pm.floatX(e0 * np.ones((1, args.n_topic))),
            beta=pm.floatX(f0 * np.ones((1, args.n_topic))), shape=(args.bsz, args.n_topic))

        gamma = Gamma("gamma", alpha=gamma0, beta=c0, shape=(args.bsz, args.n_topic))

        theta = Gamma("theta", alpha=gamma,
            beta=pm.floatX((pn / (1. - pn)) * np.ones((args.bsz, args.n_topic))),
            shape=(args.bsz, args.n_topic), total_size=args.n_tr, )

        doc = pm.DensityDist("doc", log_prob(beta, theta), observed=doc_tr)

    encoder = ThetaEncoder(n_words=args.n_word, n_hidden=100, n_topics=args.n_topic + 1)
    local_RVs = OrderedDict([(theta, encoder.encode(doc_tr))])
    encoder_params = encoder.get_params()
    s = shared(args.lr)
    def reduce_rate(a, h, i):
        s.set_value(args.lr / ((i / args.bsz) + 1) ** 0.7)

    with model:
        approx = pm.MeanField(local_rv=local_RVs)
        approx.scale_cost_to_minibatch = False
        inference = pm.KLqp(approx)

    inference.fit(args.n_iter,
                  callbacks=[reduce_rate, pm.callbacks.CheckParametersConvergence(diff="absolute")],
                  obj_optimizer=pm.adam(learning_rate=s),
                  more_obj_params=encoder_params,
                  total_grad_norm_constraint=200,
                  more_replacements={ doc_tr: doc_tr_minibatch }, )

    doc_tr.set_value(docs_tr.toarray())

    inp = tt.matrix(dtype="int64")
    sample_vi_theta = theano.function([inp],
        approx.sample_node(approx.model.theta, args.n_sample, more_replacements={doc_tr: inp}), )

    test = docs_te.toarray()
    test_n = test.sum(1)

    beta_pymc3 = pm.sample_approx(approx, draws=args.n_sample)['beta']
    theta_pymc3 = sample_vi_theta(test)

    assert beta_pymc3.shape == (args.n_sample, args.n_topic, args.n_word)
    assert theta_pymc3.shape == (args.n_sample, args.n_te, args.n_topic)

    beta_mean = beta_pymc3.mean(0)
    theta_mean = theta_pymc3.mean(0)

    pred_rate = theta_mean.dot(beta_mean)
    pp_test = (test * np.log(pred_rate) - pred_rate - sc.gammaln(test + 1)).sum(1) / test_n

    posteriors = { 'theta': theta_pymc3, 'beta': beta_pymc3,}

    log_top_words(beta_pymc3.mean(0), feature_names, n_top_words=args.n_top_word)
    save_elbo(approx.hist)
    save_pp(pp_test)
    save_draws(posteriors)


def run_dirpfa(args):
    tf_vectorizer, docs_tr, docs_te = prepare_sparse_matrix_nonlabel(args.n_tr, args.n_te, args.n_word)
    feature_names = tf_vectorizer.get_feature_names()
    doc_tr_minibatch = pm.Minibatch(docs_tr.toarray(), args.bsz)
    doc_tr = shared(docs_tr.toarray()[:args.bsz])

    def log_prob(beta, theta, n):
        """Returns the log-likelihood function for given documents.

        K : number of topics in the model
        V : number of words (size of vocabulary)
        D : number of documents (in a mini-batch)

        Parameters
        ----------
        beta : tensor (K x V)
            Word distributions.
        theta : tensor (D x K)
            Topic distributions for documents.
        n: tensor (D x 1)
            Expected lengths of each documents
        """

        def ll_docs_f(docs):
            dixs, vixs = docs.nonzero()
            vfreqs = docs[dixs, vixs]
            ll_docs = (vfreqs *
                       (pmmath.logsumexp(tt.log(theta[dixs]) + tt.log(beta.T[vixs]),
                                                  axis=1).ravel() + tt.log(
                n[dixs]).ravel()) - tt.exp(
                pmmath.logsumexp(tt.log(theta[dixs]) + tt.log(beta.T[vixs], ), axis=1)).ravel() * n[
                           dixs].ravel() - pm.distributions.special.gammaln(vfreqs + 1)

                       )
            return tt.sum(ll_docs) / (tt.sum(vfreqs) + 1e-9)

        return ll_docs_f

    with pm.Model() as model:
        n = Gamma("n",
                  alpha=pm.floatX(10. * np.ones((args.bsz, 1))),
                  beta=pm.floatX(0.1 * np.ones((args.bsz, 1))), shape=(args.bsz, 1),
                  total_size=args.n_tr)

        beta = Dirichlet("beta",
                         a=pm.floatX((1.0 / args.n_topic) * np.ones((args.n_topic, args.n_word))),
                         shape=(args.n_topic, args.n_word), )

        theta = Dirichlet("theta",
                          a=pm.floatX((1.0 / args.n_topic) * np.ones((args.bsz, args.n_topic))),
                          shape=(args.bsz, args.n_topic),
                          total_size=args.n_tr, )

        doc = pm.DensityDist("doc", log_prob(beta, theta, n), observed=doc_tr)

    encoder = ThetaNEncoder(n_words=args.n_word, n_hidden=100, n_topics=args.n_topic)
    local_RVs = OrderedDict([(theta, encoder.encode(doc_tr)[0]), (n, encoder.encode(doc_tr)[1])])
    encoder_params = encoder.get_params()
    s = shared(args.lr)
    def reduce_rate(a, h, i):
        s.set_value(args.lr / ((i / args.bsz) + 1) ** 0.7)

    with model:
        approx = pm.MeanField(local_rv=local_RVs)
        approx.scale_cost_to_minibatch = False
        inference = pm.KLqp(approx)

    inference.fit(args.n_iter,
                  callbacks=[reduce_rate, pm.callbacks.CheckParametersConvergence(diff="absolute")],
                  obj_optimizer=pm.adam(learning_rate=s), more_obj_params=encoder_params,
                  total_grad_norm_constraint=200,
                  more_replacements={ doc_tr: doc_tr_minibatch }, )

    doc_tr.set_value(docs_tr.toarray())

    inp = tt.matrix(dtype="int64")
    sample_vi_theta = theano.function([inp],
        approx.sample_node(approx.model.theta, args.n_sample, more_replacements={ doc_tr: inp
                                                                                    }), )
    sample_vi_n = theano.function([inp],
        approx.sample_node(approx.model.n, args.n_sample, more_replacements={ doc_tr: inp }))

    test = docs_te.toarray()
    test_n = test.sum(1)

    beta_pymc3 = pm.sample_approx(approx, draws=args.n_sample)['beta']
    theta_pymc3 = sample_vi_theta(test)
    n_pymc3 = sample_vi_n(test)

    assert beta_pymc3.shape == (args.n_sample, args.n_topic, args.n_word)
    assert theta_pymc3.shape == (args.n_sample, args.n_te, args.n_topic)
    assert n_pymc3.shape == (args.n_sample, args.n_te, 1)

    beta_mean = beta_pymc3.mean(0)
    theta_mean = theta_pymc3.mean(0)
    n_mean = n_pymc3.mean(0)

    pred_rate = theta_mean.dot(beta_mean) * n_mean
    pp_test = (test * np.log(pred_rate) - pred_rate - sc.gammaln(test + 1)).sum(1) / test_n

    posteriors = { 'theta': theta_pymc3, 'beta': beta_pymc3, 'n': n_pymc3}

    log_top_words(beta_pymc3.mean(0), feature_names, n_top_words=args.n_top_word)
    save_elbo(approx.hist)
    save_pp(pp_test)
    save_draws(posteriors)


run = {'dirpfa': run_dirpfa, 'pfa': run_pfa, 'lda': run_lda}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_topic', type=int, default=20)
    parser.add_argument('--n_word', type=int, default=3_000)
    parser.add_argument('--max_df', type=float, default=0.5)
    parser.add_argument('--min_df', type=int, default=10)

    parser.add_argument('--n_tr', type=int, default=10_000)
    parser.add_argument('--n_te', type=int, default=1_000)

    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--bsz', type=int, default=128)
    parser.add_argument('--n_iter', type=int, default=5000)

    parser.add_argument('--model', type=str, choices=['pfa', 'dirpfa', 'lda'])
    parser.add_argument('--n_sample', type=int, default=1000, help='posterior samples to draw')
    parser.add_argument('--n_top_word', type=int, default=10, help='Top words from each topic')

    args = parser.parse_args()

    result_path = os.path.join('result', f'{args.model}-{args.n_topic}')
    makedirs(result_path)
    logger = Logger(os.path.join(result_path, 'logs.txt'))

    def log_top_words(beta, feature_names, n_top_words=20):
        for i in range(len(beta)):
            logger.logging(("Topic #%d: " % i) + " , ".join(
                [feature_names[j] for j in beta[i].argsort()[: -n_top_words - 1: -1]]))

    def save_elbo(elbos):
        with open(os.path.join(result_path, 'elbo_val.pkl'), 'wb') as f:
            pickle.dump(elbos, f)

    def save_pp(pps):
        with open(os.path.join(result_path, 'pp.pkl'), 'wb') as f:
            pickle.dump(pps, f)

    def save_draws(draws):
        with open(os.path.join(result_path, 'draws.pkl'), 'wb') as f:
            pickle.dump(draws, f)

    run[args.model](args)

