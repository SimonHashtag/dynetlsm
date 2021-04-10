# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport log, exp, sqrt

import numpy as np
cimport numpy as np
import scipy.stats as stats


ctypedef np.npy_float64 DOUBLE
ctypedef np.npy_int64 INT

cdef inline double expit(double z):
    return 1. / (1. + exp(-z))

# TODO: Case-control sampler for directed weighted network (see directed_likelihodds_fast.pyx)
# TODO: Directed weighted network probas for ROC AUC calculation in ..._simulation.py

def directed_weighted_intercept_grad(DOUBLE[:, :, :] Y,
                                     DOUBLE[:, :, :] dist,
                                     DOUBLE[:] radii,
                                     double intercept_in,
                                     double intercept_out,
                                     double nu):
    cdef int i, j, t = 0
    cdef int n_time_steps = Y.shape[0]
    cdef int n_nodes = Y.shape[1]
    cdef double d_in, d_out, eta, step
    cdef double in_grad, out_grad = 0.
    cdef double std = sqrt(nu)

    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    d_in = (1 - dist[t, i, j] / radii[j])
                    d_out = (1 - dist[t, i, j] / radii[i])
                    eta = intercept_in * d_in + intercept_out * d_out
                    if Y[t, i, j] > 0:
                        step = (Y[t, i, j] - eta)/std**2
                    else:
                        step = (-stats.norm.pdf(eta/std)/std)/(1-stats.norm.cdf(eta/std))

                    in_grad += d_in * step
                    out_grad += d_out * step

    return np.array([in_grad, out_grad])


def directed_weighted_partial_loglikelihood(DOUBLE[:, ::1] Y,
                                            DOUBLE[:, ::1] X,
                                            DOUBLE[:] radii,
                                            double intercept_in,
                                            double intercept_out,
                                            double nu,
                                            int node_id,
                                            bint squared=False):
    cdef int j, d = 0
    cdef int n_nodes = Y.shape[0]
    cdef int n_features = X.shape[1]
    cdef double pi = 3.141592653589793
    cdef double dist = 0
    cdef double eta = 0
    cdef double loglik  = 0
    cdef double std = sqrt(nu)

    for j in range(n_nodes):
        dist = 0
        eta = 0
        if j != node_id:
            for d in range(n_features):
                dist += (X[j, d] - X[node_id, d]) ** 2

            if not squared:
                dist = sqrt(dist)

            # Y_ijt
            eta = intercept_in * (1 - dist / radii[j]) + intercept_out * (1 - dist / radii[node_id])
            if Y[node_id, j] > 0:
                loglik += -log(std) - 1/2 * log(2*pi) - 1/2*((Y[node_id, j] - eta)/std)**2
            else:
                loglik += log(1 - stats.norm.cdf(eta/std))

            # Y_jit
            eta = intercept_in * (1 - dist / radii[node_id]) + intercept_out * (1 - dist / radii[j])
            if Y[node_id, j] > 0:
                loglik += -log(std) - 1/2 * log(2*pi) - 1/2*((Y[j, node_id] - eta)/std)**2
            else:
                loglik += log(1 - stats.norm.cdf(eta/std))

    return loglik


def directed_weighted_network_loglikelihood_fast(DOUBLE[:, :, ::1] Y,
                                                 DOUBLE[:, :, ::1] dist,
                                                 DOUBLE[:] radii,
                                                 double intercept_in,
                                                 double intercept_out,
                                                 double nu):
    cdef int i, j, t = 0
    cdef int n_time_steps = Y.shape[0]
    cdef int n_nodes = Y.shape[1]
    cdef double pi = 3.141592653589793
    cdef double d_in, d_out, eta = 0.
    cdef double loglik = 0.
    cdef double std = sqrt(nu)

    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    d_in = (1 - dist[t, i, j] / radii[j])
                    d_out = (1 - dist[t, i, j] / radii[i])
                    eta = intercept_in * d_in + intercept_out * d_out
                    if Y[t, i, j] > 0:
                        loglik += -log(std) - 1/2 * log(2*pi) - 1/2*((Y[t, i, j] - eta)/std)**2
                    else:
                        loglik += log(1 - stats.norm.cdf(eta/std))

    return loglik
