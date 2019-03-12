#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:20:11 2018
@author: Paris
"""

import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import timeit


class RNN:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):

        # Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        # X has the form lags x data x dim
        # Y has the form data x dim

        # Define PyTorch variables
        X = torch.from_numpy(X).type(self.dtype)
        Y = torch.from_numpy(Y).type(self.dtype)
        self.X = Variable(X, requires_grad=False)
        self.Y = Variable(Y, requires_grad=False)

        self.X_dim = X.shape[-1]
        self.Y_dim = Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.lags = X.shape[0]

        # Initialize network weights and biases
        self.U, self.b, self.W, self.V, self.c = self.initialize_RNN()

        # Store loss values
        self.training_loss = []

        # Define optimizer
        self.optimizer = torch.optim.Adam([self.U, self.b, self.W, self.V, self.c], lr=1e-3)

    # Initialize network weights and biases using Xavier initialization
    def initialize_RNN(self):
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
            return Variable(xavier_stddev * torch.randn(in_dim, out_dim).type(self.dtype), requires_grad=True)

        U = xavier_init(size=[self.X_dim, self.hidden_dim])
        b = Variable(torch.zeros(1, self.hidden_dim).type(self.dtype), requires_grad=True)

        W = Variable(torch.eye(self.hidden_dim).type(self.dtype), requires_grad=True)

        V = xavier_init(size=[self.hidden_dim, self.Y_dim])
        c = Variable(torch.zeros(1, self.Y_dim).type(self.dtype), requires_grad=True)

        return U, b, W, V, c

    # Evaluates the forward pass
    def forward_pass(self, X):
        H = torch.zeros(X.shape[1], self.hidden_dim).type(self.dtype)
        for i in range(0, self.lags):
            H = F.tanh(torch.matmul(H, self.W) + torch.matmul(X[i, :, :], self.U) + self.b)
        H = torch.matmul(H, self.V) + self.c
        return H

    # Computes the mean square error loss
    def compute_loss(self, X, Y):
        loss = torch.mean((Y - self.forward_pass(X)) ** 2)
        return loss

    # Fetches a mini-batch of data
    def fetch_minibatch(self, X, y, N_batch):
        N = X.shape[1]
        idx = torch.randperm(N)[0:N_batch]
        X_batch = X[:, idx, :]
        y_batch = y[idx, :]
        return X_batch, y_batch

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=100):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch mini-batch
            X_batch, Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)

            loss = self.compute_loss(X_batch, Y_batch)

            # Store loss value
            self.training_loss.append(loss)

            # Backward pass
            loss.backward()

            # update parameters
            self.optimizer.step()

            # Reset gradients for next step
            self.optimizer.zero_grad()

            # Print
            if it % 50 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss.cpu().data.numpy(), elapsed))
                start_time = timeit.default_timer()

    # Evaluates predictions at test points
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).type(self.dtype)
        y_star = self.forward_pass(X_star)
        y_star = y_star.cpu().data.numpy()
        return y_star


# Define CNN architecture and forward pass
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.fc = torch.nn.Linear(7 * 7 * 32, 10)

    def forward_pass(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ConvNet:
    # Initialize the class
    def __init__(self, X, Y):

        # Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor

        # Define PyTorch dataset
        X = torch.from_numpy(X).type(self.dtype_double)  # num_images x num_pixels_x x num_pixels_y
        Y = torch.from_numpy(Y).type(self.dtype_int)  # num_images x 1
        self.train_data = torch.utils.data.TensorDataset(X, Y)

        # Define architecture and initialize
        self.net = CNN()

        # Define the loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    # Trains the model by minimizing the Cross Entropy loss
    def train(self, num_epochs=10, batch_size=128):

        # Create a PyTorch data loader object
        self.trainloader = torch.utils.data.DataLoader(self.train_data,
                                                       batch_size=batch_size,
                                                       shuffle=True)

        start_time = timeit.default_timer()
        for epoch in range(num_epochs):
            for it, (images, labels) in enumerate(self.trainloader):
                images = Variable(images)
                labels = Variable(labels)

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.net.forward_pass(images)

                # Compute loss
                loss = self.loss_fn(outputs, labels)

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimizer.step()

                if (it + 1) % 100 == 0:
                    elapsed = timeit.default_timer() - start_time
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Time: %2fs'
                          % (
                          epoch + 1, num_epochs, it + 1, len(self.train_data) // batch_size, loss.cpu().data, elapsed))
                    start_time = timeit.default_timer()

    def test(self, X, Y):
        # Define PyTorch dataset
        X = torch.from_numpy(X).type(self.dtype_double)  # num_images x num_pixels_x x num_pixels_y
        Y = torch.from_numpy(Y).type(self.dtype_int)  # num_images x 1
        test_data = torch.utils.data.TensorDataset(X, Y)

        # Create a PyTorch data loader object
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=128,
                                                  shuffle=True)

        # Test prediction accuracy
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images)
            outputs = self.net.forward_pass(images)
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Test Accuracy of the model on the %d test images: %.5f %%' % (len(test_data), 100.0 * correct / total))

    # Evaluates predictions at test points
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).type(self.dtype_double)
        X_star = Variable(X_star, requires_grad=False)
        y_star = self.net.forward_pass(X_star)
        y_star = y_star.cpu().data.numpy()
        return y_star


