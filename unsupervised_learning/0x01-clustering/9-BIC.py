#!/usr/bin/env python3
""" BIC """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ finds the best number of clusters for a GMM using the Bayesian
        Information Criterion
        - X is a numpy.ndarray of shape (n, d) containing the data set
        - kmin is a positive integer containing the minimum number of
          clusters to check for (inclusive)
        - kmax is a positive integer containing the maximum number of
          clusters to check for (inclusive)
            - If kmax is None, kmax should be set to the maximum number
              of clusters possible
        - iterations is a positive integer containing the maximum number
          of iterations for the EM algorithm
        - tol is a non-negative float containing the tolerance for the
          EM algorithm
        - verbose is a boolean that determines if the EM algorithm
          should print information to the standard output
        Returns: best_k, best_result, l, b, or None, None, None, None
                 on failure
            - best_k is the best value for k based on its BIC
            - best_result is tuple containing pi, m, S
                - pi is a numpy.ndarray of shape (k,) containing the
                  cluster priors for the best number of clusters
                - m is a numpy.ndarray of shape (k, d) containing the
                  centroid means for the best number of clusters
                - S is a numpy.ndarray of shape (k, d, d) containing
                  the covariance matrices for the best number of clusters
            - l is a numpy.ndarray of shape (kmax - kmin + 1) containing
              the log likelihood for each cluster size tested
            - b is a numpy.ndarray of shape (kmax - kmin + 1) containing
              the BIC value for each cluster size tested
            - Use: BIC = p * ln(n) - 2 * l
                - p is the number of parameters required for the model
                - n is the number of data points used to create the model
                - l is the log likelihood of the model
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin < 1:
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if type(kmax) is not int or kmax < 1 or kmax <= kmin:
        return None, None, None, None
    if type(iterations) is not int or iterations < 1:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None
    pi_all = []
    m_all = []
    S_all = []
    l_all = []
    b = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, like = expectation_maximization(X, k, iterations,
                                                     tol, verbose)
        pi_all.append(pi)
        m_all.append(m)
        S_all.append(S)
        l_all.append(like)
        p = k * d
        b.append(p * np.log(n) - 2 * like)
    pi_all = np.array(pi_all)
    m_all = np.array(m_all)
    S_all = np.array(S_all)
    l_all = np.array(l_all)
    b = np.array(b)
    best = np.argmin(b)
    best_result = (pi_all[best], m_all[best], S_all[best])
    return best, best_result, l_all, b
