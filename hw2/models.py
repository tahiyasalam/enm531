#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:34:25 2018
@author: paris
"""

import numpy as np


class BayesianLinearRegression:
    """
      Linear regression model: y = (w.T)*x + \epsilon
      w ~ N(0,beta^(-1)I)
      P(y|x,w) ~ N(y|(w.T)*x,alpha^(-1)I)
    """

    def __init__(self, X, y, alpha=1.0, beta=1.0):
        self.X = X
        self.y = y

        self.alpha = alpha
        self.beta = beta

        self.jitter = 1e-8

    def fit_MLE(self):
        xTx_inv = np.linalg.inv(np.matmul(self.X.T, self.X) + self.jitter)
        xTy = np.matmul(self.X.T, self.y)
        w_MLE = np.matmul(xTx_inv, xTy)

        self.w_MLE = w_MLE

        return w_MLE

    def fit_MAP(self):
        Lambda = np.matmul(self.X.T, self.X) + \
                 (self.beta / self.alpha) * np.eye(self.X.shape[1])
        Lambda_inv = np.linalg.inv(Lambda)
        xTy = np.matmul(self.X.T, self.y)
        mu = np.matmul(Lambda_inv, xTy)

        self.w_MAP = mu
        self.Lambda_inv = Lambda_inv

        return mu, Lambda_inv

    def predictive_distribution(self, X_star):
        mean_star = np.matmul(X_star, self.w_MAP)
        var_star = 1.0 / self.alpha + \
                   np.matmul(X_star, np.matmul(self.Lambda_inv, X_star.T))
        return mean_star, var_star

    def fit_MLE_monomial(self, M):
        # Size of phi: d x M
        Phi = np.zeros((np.shape(self.X)[0], (M + 1)))
        for c in range(0, np.shape(Phi)[1]):
            Phi[:,c] = np.power(self.X[:,0].T, c)
        phiTphi_inv = np.linalg.inv(np.matmul(Phi.T, Phi) + self.jitter)
        phiTy = np.matmul(Phi.T, self.y)
        w_MLE = np.matmul(phiTphi_inv, phiTy)

        self.w_MLE = w_MLE

        return w_MLE

    def fit_MAP_monomial(self, M):
        # Size of phi: d x M
        Phi = np.zeros((np.shape(self.X)[0], (M + 1)))
        print(np.shape(Phi))
        for c in range(0, np.shape(Phi)[1]):
            Phi[:,c] = np.power(self.X[:,0].T, c)
        Lambda = np.matmul(Phi.T, Phi) + \
                 (self.beta / self.alpha) * np.eye(Phi.shape[1])
        Lambda_inv = np.linalg.inv(Lambda)
        phiTy = np.matmul(Phi.T, self.y)
        mu = np.matmul(Lambda_inv, phiTy)

        self.w_MAP = mu
        self.Lambda_inv = Lambda_inv

        return mu, Lambda_inv

    def fit_MLE_basis(self, type, M):
        # Size of phi: d x M

        Phi = self.basis(type, M)

        phiTphi_inv = np.linalg.inv(np.matmul(Phi.T, Phi) + self.jitter)
        phiTy = np.matmul(Phi.T, self.y)
        w_MLE = np.matmul(phiTphi_inv, phiTy)

        self.w_MLE = w_MLE

        return w_MLE

    def fit_MAP_basis(self, type, M):
        # Size of phi: d x M
        Phi = self.basis(type, M)
        Lambda = np.matmul(Phi.T, Phi) + \
                 (self.beta / self.alpha) * np.eye(Phi.shape[1])
        Lambda_inv = np.linalg.inv(Lambda)
        phiTy = np.matmul(Phi.T, self.y)
        mu = np.matmul(Lambda_inv, phiTy)

        self.w_MAP = mu
        self.Lambda_inv = Lambda_inv

        return mu, Lambda_inv

    def basis(self, type, M, X_star = None):
        if type is 'monomial':
            if X_star is None:
                Phi = np.zeros((np.shape(self.X)[0], (M + 1)))
                for c in range(0, np.shape(Phi)[1]):
                    Phi[:, c] = np.power(self.X[:, 0].T**c)
            else:
                Phi = np.zeros((np.shape(X_star)[0], (M + 1)))
                for c in range(0, np.shape(Phi)[1]):
                    Phi[:, c] = np.power(X_star[:, 0].T, c)
        if type is 'fourier':
            if X_star is None:
                Phi = np.zeros((np.shape(self.X)[0], (M + 1)*2))
                for c in range(0, np.shape(Phi)[1], 2):
                    Phi[:, c] = np.sin(c*np.pi * self.X[:, 0].T)
                    Phi[:, c+1] = np.cos(c*np.pi * self.X[:, 0].T)
            else:
                Phi = np.zeros((np.shape(X_star)[0], (M + 1)*2))
                for c in range(0, np.shape(Phi)[1], 2):
                    Phi[:, c] = np.sin(c * np.pi * X_star[:, 0].T)
                    Phi[:, c+1] = np.cos(c * np.pi * X_star[:, 0].T)
        if type is 'legendre':
            from numpy.polynomial import Legendre
            if X_star is None:
                Phi = Legendre.basis(M)(self.X)
            else:
                Phi = Legendre.basis(M)(X_star)

        return Phi