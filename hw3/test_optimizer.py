#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tahiyasalam
"""

import autograd.numpy as np
from autograd import elementwise_grad, grad
import matplotlib.pyplot as plt
from pyDOE import lhs

from optimizer import AdamOptimizer

'''
MLE estimation for linear regression model with M Fourier features and Gaussian likelihood
'''


def estimate_function(x):
    '''
    target function y(x) = 2*sin(2*pi*x) + sin(8*pi*x) + 0.5*sin(16*pi*x), x in [-1, 1]

    :param x: discretized domain
    :return: set of y for discretized domain
    '''
    return 2 * np.sin(2 * np.pi * x) + np.sin(8 * np.pi * x) + 0.5 * np.sin(16 * np.pi * x)


def loss_function_MLE(w, Phi, y):
    '''

    Loss function for MLE with arbitrary features

    :param w: weight vectors
    :param Phi: feature matrix
    :param y: target predictions
    :return: loss function for MLE
    '''
    return 1/2*y.T@y - y.T@Phi@w + 1/2*w.T@Phi.T@Phi@w


if __name__ == "__main__":
    N = 500
    M = 16
    learning_rate = 1e-3
    iter_num = 10000
    noise_var = 0.2

    # Specify batch size for mini-batch
    batch_size = N

    x_min = -1
    x_max = 1

    # Create random input and output data
    X = lhs(1, N) * (x_max - x_min) + x_min
    y = estimate_function(X)
    y_noisy =  y + np.std(y) * noise_var * np.random.randn(N, 1)

    ao = AdamOptimizer()
    ao.learning_rate = learning_rate

    w_MLE = np.random.randn((M + 1)*2)
    Phi = ao.basis('fourier', M, X)

    m_curr = 0
    v_curr = 0

    grad_loss_array = []
    y_pred_e = []

    y_pred_MLE = np.matmul(Phi, w_MLE)  # Keep track of current prediction of y

    grad_loss_fnc = elementwise_grad(loss_function_MLE, 0)

    y_batch = y_noisy
    Phi_batch = Phi
    for n in range(1, iter_num+1):
        if batch_size < N: # Selects random elements from noisy set of data if not full batch
            rndm_idx = np.random.choice(np.shape(y_noisy)[0], size=batch_size, replace=False)
            y_batch = y_noisy[rndm_idx]
            Phi_batch = Phi[rndm_idx,:]
        grad_loss = grad_loss_fnc(w_MLE, Phi_batch, y_batch)  # Compute gradient of loss function

        grad_loss_array.append(loss_function_MLE(w_MLE, Phi_batch, y_batch).sum())

        # Apply gradient and update weight vector and mean and variance estimates
        w_MLE, m_curr, v_curr = ao.apply_gradient(grad_loss, w_MLE, m_curr, v_curr, batch_size, n)
        y_pred_MLE = np.matmul(Phi, w_MLE)

    y_pred_MLE = np.matmul(Phi, w_MLE)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(X, y_noisy, 'o', label='Noisy data')
    plt.plot(X, y_pred_MLE, 'o', label='Prediction')
    plt.plot(X, y, 'o', label='Actual function', markersize=0.75)
    plt.legend()
    plt.title('Adam Optimizer estimating data for batch size N = ' + str(batch_size) + ' LR = ' + str(learning_rate))
    plt.legend()

    plt.subplot(212)
    plt.plot(range(1, len(grad_loss_array)+1), grad_loss_array, 'o', label='loss')
    plt.ylabel('loss')
    plt.xlabel('iteration number')
    plt.title('Loss function for N = ' + str(batch_size) + ' LR = ' + str(learning_rate))
    plt.legend()
    plt.show()







