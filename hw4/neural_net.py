#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tahiyasalam
"""
import numpy as np
from pyDOE import lhs
import autograd.numpy as np
from autograd import grad, elementwise_grad
from optimizer import AdamOptimizer
import matplotlib.pyplot as plt


def f(x, y):
    return np.cos(np.pi*x) * np.cos(np.pi*y)


def init_params(params):  # Xavier initialization of data
    for nl in range(0, len(params)):
        d_in = np.shape(params[nl][0])[0]
        d_out = np.shape(params[nl][0])[1]

        mean = 0
        var = 2/(d_in+d_out)
        params[nl][0] = np.random.normal(mean, var, np.shape(params[nl][0]))  # Weights selected from normal distribution

    return params


def feed_forward(params, x):
    [w1, b1], [w2, b2], [wo, bo] = params
    h1 = np.tanh(np.matmul(x, w1) + b1)  # Output from first layer
    h2 = np.tanh(np.matmul(h1, w2) + b2)  # Output from second layer
    ho = np.tanh(np.matmul(h2, wo) + bo)  # Output layer
    # print(np.shape(h2), np.shape(wo), np.shape(ho))
    return ho


def objective(params, z, X):
    pred = feed_forward(params, X)
    err = z.reshape([-1, 1]) - pred
    return np.mean(err**2)


def calc_l2(z, z_pred):
    return np.linalg.norm(z-z_pred)


if __name__ == "__main__":
    num_sample = []
    error = []
    for i in range(10, 501, 10):
        N = i

        batch_size = N
        hidden_layers = 2
        neurons = 50

        min = 50
        max = 54

        # Create random input and output data
        x = lhs(1, N) * (max - min) + min
        y = lhs(1, N) * (max - min) + min

        X = np.hstack((x, y))  # X has dimension (N, D)
        D = np.shape(X)[1]
        W_1 = np.zeros((D, neurons))
        W_2 = np.zeros((np.shape(W_1)[1], neurons))
        W_o = np.zeros((np.shape(W_2)[1], 1))
        b_1 = np.zeros(np.shape(y))
        b_2 = np.zeros(np.shape(y))
        b_o = np.zeros(np.shape(y))

        # Output layer?

        params = [[W_1, b_1], [W_2, b_2], [W_o, b_o]]
        params = init_params(params)
        iter_num = int(1e4)

        noise_var = 0.05
        z = f(x, y) + noise_var * np.random.randn(N, 1)

        new_params = [[None, None], [None, None], [None, None]]
        mean_var = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]

        ao = AdamOptimizer()
        for n in range(1, iter_num+1):
            grad_loss = grad(objective, 0)
            grad_loss = grad_loss(params, z, X)
            for nl in range(0, len(new_params)):
                for nw in range(0, len(new_params[nl])):
                    new_params[nl][nw], mean_var[nl][nw][0], mean_var[nl][nw][1] = ao.apply_gradient(grad_loss[nl][nw], params[nl][nw], mean_var[nl][nw][0], mean_var[nl][nw][1], batch_size, n)
            params = new_params

        z_pred = feed_forward(params, X)

        num_sample.append(i)
        error.append(calc_l2(z_pred, z))

    plt.scatter(num_sample, error)
    plt.xlabel('Number of samples')
    plt.ylabel('L2 error between predicted and actual')
    plt.title('Performance of neural net with 2 hidden layers and 50 neurons per layer')
    plt.show()