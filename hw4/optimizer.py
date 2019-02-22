#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tahiyasalam
"""

import autograd.numpy as np
from autograd import grad, elementwise_grad


class AdamOptimizer:
    """
      Linear regression model: y = (w.T)*x + \epsilon
      w ~ N(0,beta^(-1)I)
      P(y|x,w) ~ N(y|(w.T)*x,alpha^(-1)I)
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.jitter = 1e-8

    def compute_gradient(self, loss, params=None):
        if params:
            return grad(loss, params)
        else:
            return grad(loss)

    def apply_gradient(self, grad_loss, theta_prev, m_prev, v_prev, batch_size, n):
        m_curr = self.beta1*m_prev + (1-self.beta1)*grad_loss
        v_curr = self.beta2*v_prev + (1-self.beta2)*grad_loss**2

        m_corr = m_curr/(1 - self.beta1**n)
        v_corr = v_curr/(1 - self.beta2**n)

        theta_curr = theta_prev - self.learning_rate*m_corr/(np.sqrt(v_corr) + self.jitter)

        return theta_curr, m_curr, v_curr

    def minimize(self, loss):
        self.compute_gradient()
        self.apply_gradient()

    def basis(self, type, M, X_star = None):
        if type is 'monomial':
            if X_star is None:
                Phi = np.zeros((np.shape(self.X)[0], (M + 1)))
                for c in range(0, np.shape(Phi)[1]):
                    Phi[:, c] = self.X[:, 0].T**c
            else:
                Phi = np.zeros((np.shape(X_star)[0], (M + 1)))
                for c in range(0, np.shape(Phi)[1]):
                    Phi[:, c] = X_star[:, 0].T**c
        if type is 'fourier':
            if X_star is None:
                Phi = np.zeros((np.shape(self.X)[0], (M + 1)*2))
                for c in range(0, np.shape(Phi)[1], 2):
                    Phi[:, c] = np.sin(int(c/2) * np.pi * self.X[:, 0].T)
                    Phi[:, c+1] = np.cos(int(c/2) * np.pi * self.X[:, 0].T)
            else:
                Phi = np.zeros((np.shape(X_star)[0], (M + 1)*2))
                for c in range(0, np.shape(Phi)[1], 2):
                    Phi[:, c] = np.sin(int(c/2) * np.pi * X_star[:, 0].T)
                    Phi[:, c+1] = np.cos(int(c/2) * np.pi * X_star[:, 0].T)
        if type is 'legendre':
            from numpy.polynomial import Legendre
            if X_star is None:
                Phi = Legendre.basis(M)(self.X)
            else:
                Phi = Legendre.basis(M)(X_star)
        return Phi