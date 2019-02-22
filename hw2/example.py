#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:35:11 2018

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm

from models import BayesianLinearRegression

if __name__ == "__main__":

    # N is the number of training points.
    N = 500
    M = 16
    noise_var = 0.5
    alpha = 5
    beta = 0.1

    x_max = 2

    # Create random input and output data
    X = lhs(1, N) * x_max
    y = np.exp(X) * np.sin(2*np.pi*X) + noise_var * np.random.randn(N, 1)

    # Define model
    m = BayesianLinearRegression(X, y, alpha, beta)

    # Fit MLE and MAP estimates for w
    w_MLE = m.fit_MLE()
    w_MAP, Lambda_inv = m.fit_MAP()

    # Predict at a set of test points
    X_star = np.linspace(0, x_max, 200)[:, None]

    y_pred_MLE = np.matmul(X_star, w_MLE)
    y_pred_MAP = np.matmul(X_star, w_MAP)

    # Draw samples from the predictive posterior
    num_samples = 500
    mean_star, var_star = m.predictive_distribution(X_star)
    samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)

    # Plot
    plt.figure(1, figsize=(8, 6))
    for i in range(0, num_samples):
        plt.plot(X_star, samples[i, :], 'k', linewidth=0.05)
    plt.plot(X_star, y_pred_MLE, linewidth=3.0, label='MLE for Identity')
    plt.plot(X_star, y_pred_MAP, linewidth=3.0, label='MAP for Identity')
    plt.plot(X, y, 'o', label='Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Bayesian Linear Regression with Identity Basis (M = ' + str(M) + ', N = ' + str(N) + ')')
    plt.savefig('identity_basis_' + 'M_' + str(M) + '_N_' + str(N) + '.png')

    #    plt.axis('tight')

    # # Monomial
    plt.clf()
    w_MLE = m.fit_MLE_basis('monomial', M)
    w_MAP, Lambda_inv = m.fit_MAP_basis('monomial', M)
    Phi = m.basis('monomial', M, X_star)

    y_pred_MLE = np.matmul(Phi, w_MLE)
    y_pred_MAP = np.matmul(Phi, w_MAP)

    # Draw sampes from the predictive posterior
    num_samples = 500
    mean_star, var_star = m.predictive_distribution(Phi)
    samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)

    # Plot
    plt.figure(1, figsize=(8, 6))
    for i in range(0, num_samples):
        plt.plot(X_star, samples[i, :], 'k', linewidth=0.05)
    plt.plot(X_star, y_pred_MLE, linewidth=3.0, label='MLE for Monomial with M = ' + str(M))
    plt.plot(X_star, y_pred_MAP, linewidth=3.0, label='MAP for Monomial with M = ' + str(M))
    plt.plot(X, y, 'o', label='Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Bayesian Linear Regression with Monomial Basis (M = ' + str(M) + ', N = ' + str(N) + ')')
    plt.savefig('monomial_basis_' + 'M_' + str(M) + '_N_' + str(N) + '.png')
    #    plt.axis('tight')

    # # Fourier
    plt.clf()

    w_MLE = m.fit_MLE_basis('fourier', M)
    w_MAP, Lambda_inv = m.fit_MAP_basis('fourier', M)
    Phi = m.basis('fourier', M, X_star)

    y_pred_MLE = np.matmul(Phi, w_MLE)
    y_pred_MAP = np.matmul(Phi, w_MAP)

    # Draw sampes from the predictive posterior
    num_samples = 500
    mean_star, var_star = m.predictive_distribution(Phi)
    samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)

    # Plot
    plt.figure(1, figsize=(8, 6))
    for i in range(0, num_samples):
        plt.plot(X_star, samples[i, :], 'k', linewidth=0.05)
    plt.plot(X_star, y_pred_MLE, linewidth=3.0, label='MLE for Fourier')
    plt.plot(X_star, y_pred_MAP, linewidth=3.0, label='MAP for Fourier')
    plt.plot(X, y, 'o', label='Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Bayesian Linear Regression with Fourier Basis (M = ' + str(M) + ', N = ' + str(N) + ')')
    plt.savefig('fourier_basis_' + 'M_' + str(M) + '_N_' + str(N) + '.png')

    # Legendre
    plt.clf()

    w_MLE = m.fit_MLE_basis('legendre', M)
    w_MAP, Lambda_inv = m.fit_MAP_basis('legendre', M)
    Phi = m.basis('legendre', M, X_star)

    y_pred_MLE = np.matmul(Phi, w_MLE)
    y_pred_MAP = np.matmul(Phi, w_MAP)

    # Draw sampes from the predictive posterior
    num_samples = 500
    mean_star, var_star = m.predictive_distribution(Phi)
    samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)

    # Plot
    plt.figure(1, figsize=(8, 6))
    for i in range(0, num_samples):
        plt.plot(X_star, samples[i, :], 'k', linewidth=0.05)
    plt.plot(X_star, y_pred_MLE, linewidth=3.0, label='MLE for Legendre')
    plt.plot(X_star, y_pred_MAP, linewidth=3.0, label='MAP for Legendre')
    plt.plot(X, y, 'o', label='Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Bayesian Linear Regression with Legendre Basis (M = ' + str(M) + ', N = ' + str(N) + ')')
    plt.savefig('legendre_basis_' + 'M_' + str(M) + '_N_' + str(N) + '.png')

plt.show()