#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tahiyasalam
"""

import autograd.numpy as np
from autograd import elementwise_grad, grad
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

from optimizer import AdamOptimizer

"""
Binary cross entropy for logistic regression model
"""


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def loss_function_binary_cross_entropy(w, x, y):
    a = sigmoid(np.dot(x, w))
    prob = np.log(a) * y + np.log(1 - a) * (1 - y)
    return -np.sum(prob)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == "__main__":
    N = 10000
    M = 12
    learning_rate = 1e-3
    iter_num = 20000

    batch_size = 32

    training_size = int(N * (3/3))

    X = np.zeros((N, 12))

    csv_file = 'Data_for_UCI_named.csv'
    X_0 = np.genfromtxt(csv_file, delimiter=',')

    X = X_0[1:, 0:12]
    Y = np.zeros(np.shape(X)[0])

    Y_0 = np.genfromtxt(csv_file, delimiter=',', usecols=(-1), dtype=np.str)[1:]

    bad_chars = '"'''
    for i in range(0, np.shape(Y_0)[0]):
        s = Y_0[i]
        for c in bad_chars: s = s.replace(c, "")
        if s == "unstable":
            Y[i] = 0
        else:
            Y[i] = 1

    X_tr = X[0:training_size,:]
    Y_tr = Y[0:training_size]

    w_star = np.zeros(M)

    ao = AdamOptimizer()
    grad_loss_fnc = elementwise_grad(loss_function_binary_cross_entropy, 0)
    grad_loss_array = []

    m_curr = 0
    v_curr = 0
    Y_batch = Y_tr
    X_batch = X_tr
    for n in range(1, iter_num+1):
        if batch_size < training_size:  # Selects random elements from noisy set of data if not full batch
            rndm_idx = np.random.choice(np.shape(Y_tr)[0], size=batch_size, replace=False)
            Y_batch = Y_tr[rndm_idx]
            X_batch = X_tr[rndm_idx, :]

        grad_loss = grad_loss_fnc(w_star, X_batch, Y_batch)  # Compute gradient of loss function
        grad_loss_array.append(loss_function_binary_cross_entropy(w_star, X_batch, Y_batch).sum())

        # Apply gradient and update weight vector and mean and variance estimates
        w_star, m_curr, v_curr = ao.apply_gradient(grad_loss, w_star, m_curr, v_curr, batch_size, n)

    y_pred = sigmoid(np.dot(X, w_star))
    unstable_idx = y_pred <= 0.5
    stable_idx = y_pred > 0.5
    y_pred[unstable_idx] = 0
    y_pred[stable_idx] = 1

    cnf_matrix = confusion_matrix(Y, y_pred)

    plt.subplot(211)
    plot_confusion_matrix(cnf_matrix, normalize=False, classes=['unstable', 'stable'], title='Not normalized confusion matrix')

    plt.subplot(212)
    plt.plot(range(1, len(grad_loss_array)+1), grad_loss_array, 'o', label='loss')
    plt.ylabel('loss')
    plt.xlabel('iteration number')
    plt.title('Logistic Regression loss function for N = ' + str(batch_size) + ' LR = ' + str(learning_rate))
    plt.legend()
    plt.show()

