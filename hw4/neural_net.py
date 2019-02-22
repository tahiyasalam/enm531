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
    # [w1, b1], [w2, b2] = params

    h1 = np.tanh(np.matmul(x, w1) + b1)  # Output from first layer
    h2 = np.tanh(np.matmul(h1, w2) + b2)  # Output from second layer
    ho = np.tanh(np.matmul(h2, wo) + bo)  # Output layer
    # print(np.shape(h2), np.shape(wo), np.shape(ho))
    return ho


def feed_forward(params, x):
    h_i = x
    for i in range(0, len(params) - 1):
        [w_i, b_i] = params[i]
        print(np.shape(h_i), np.shape(w_i), np.shape(b_i))
        h_i = np.tanh(np.matmul(h_i, w_i) + b_i)  # Output from first layer

    [w_o, b_o] = params[-1]
    h_o = np.tanh(np.matmul(h_i, w_o) + b_o)  # Output layer
    return h_o


def composite_mse_loss(params, u_i, u_x, f_x, N_u, N_f):
    u_pred_u = feed_forward(params, u_x)
    err_u = u_i.reshape([-1, 1]) - u_pred_u

    u_pred_f = feed_forward(params, f_x)
    u_pred_f_x = elementwise_grad(feed_forward, 1)
    u_pred_f_xx = elementwise_grad(u_pred_f_x, 1)(params, f_x)
    err_f = u_pred_f_xx - u_pred_f - f_x

    # print(np.shape(err_f), np.shape(u_pred_f_x))
    return 1/N_u * np.sum(err_u**2) + 1/N_f * np.sum(err_f**2)


def objective(params, z, X):
    pred = feed_forward(params, X)
    err = z.reshape([-1, 1]) - pred
    return np.mean(err**2)


def calc_l2_error(z, z_pred):
    return np.linalg.norm(z-z_pred)/np.linalg.norm(z)


def problem_1():
    def f(x0, y0):
        return np.cos(np.pi * x0) * np.cos(np.pi * y0)

    num_sample = []
    error = []

    N_max = 100

    min = 50
    max = 54

    # Create random input and output data
    x_prime = lhs(1, N_max) * (max - min) + min
    y_prime = lhs(1, N_max) * (max - min) + min

    noise_var = 0.05
    z_prime = f(x_prime, y_prime) + noise_var * np.random.randn(N_max, 1)

    for i in range(10, N_max+1, 10):
        N = i

        batch_size = N
        hidden_layers = 2
        neurons = 50

        # Create random input and output data
        x = x_prime[0:i, :]
        y = y_prime[0:i, :]
        z = z_prime[0:i, :]

        X = np.hstack((x, y))  # X has dimension (N, D)
        D = np.shape(X)[1]
        W_1 = np.zeros((D, neurons))
        W_2 = np.zeros((np.shape(W_1)[1], neurons))
        W_o = np.zeros((np.shape(W_2)[1], 1))
        b_1 = np.zeros(np.shape(y))
        b_2 = np.zeros(np.shape(y))
        b_o = np.zeros(np.shape(y))

        params = [[W_1, b_1], [W_2, b_2], [W_o, b_o]]
        # params = [[W_1, b_1], [W_2, b_2]]

        params = init_params(params)
        iter_num = int(10000)


        new_params = [[None, None], [None, None], [None, None]]
        mean_var = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        # new_params = [[None, None], [None, None]]
        # mean_var = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]

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
        error.append(calc_l2_error(z, z_pred))

    print(error)
    plt.scatter(num_sample, error)
    plt.xlabel('Number of samples')
    plt.ylabel('L2 error between predicted and actual')
    plt.title('Performance of neural net with 2 hidden layers and 50 neurons per layer')
    plt.show()


def problem_2():
    N_f = 10
    batch_size = N_f

    hidden_layers = 2
    neurons = 50

    ''' f(x) = -(np.pi**2 + lambda)*np.sin(np.pi*x)'''
    def f(x, l):
        return -(np.pi**2 + l) * np.sin(np.pi*x)

    ''' u(x) = np.sin(np.pi*x)'''
    def u(x):
        return np.sin(np.pi*x)

    min = -1
    max = 1

    # Create random input and output data
    x_f = lhs(1, N_f) * (max - min) + min
    noise_var = 0.05
    z_f = f(x_f, 1) + noise_var * np.random.randn(N_f, 1)
    D = np.shape(x_f)[1]

    N_u = 2
    x_u = np.array([[-1], [1]])
    z_u = u(x_u)

    params = []
    new_params = [[[None, None]] * (hidden_layers+1)][0]
    mean_var = [[[[0, 0], [0, 0]]] * (hidden_layers+1)][0]
    for i in range(0, hidden_layers):
        if not params:
            W_i = np.zeros((D, neurons))
        else:
            W_i = np.zeros((np.shape(params[-1][0])[1], neurons))
        b_i = np.zeros((neurons, 1)).T
        # b_i = np.zeros_like(x_f).T
        params.append([W_i, b_i])

    W_o = np.zeros((np.shape(params[-1][0])[1], 1))
    b_o = np.zeros((neurons, 1)).T
    # b_o = np.zeros_like(x_f)

    params.append([W_o, b_o])
    init_params(params)

    iter_num = 1000
    ao = AdamOptimizer()
    for n in range(1, iter_num + 1):
        grad_loss = grad(composite_mse_loss, 0)
        grad_loss = grad_loss(params, z_u, x_u, x_f, N_u, N_f)
        for nl in range(0, len(new_params)):
            print('inside loop ', nl)
            for nw in range(0, len(new_params[nl])):
                print(np.shape(mean_var[nl][nw][0]), np.shape(mean_var[nl][nw][1]), np.shape(grad_loss[nl][nw]))
                new_params[nl][nw], mean_var[nl][nw][0], mean_var[nl][nw][1] = ao.apply_gradient(grad_loss[nl][nw],
                                                                                                 params[nl][nw],
                                                                                                 mean_var[nl][nw][0],
                                                                                                 mean_var[nl][nw][1],
                                                                                                 batch_size, n)
        params = new_params

    z_pred = feed_forward(params, z_f)


if __name__ == "__main__":
    # problem_1()
    problem_2()

