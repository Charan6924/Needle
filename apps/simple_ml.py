"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    with gzip.open(image_filename, 'rb') as f:
        f.read(4)  # magic
        num = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols).astype(np.float32) / 255.0

    with gzip.open(label_filename, 'rb') as f:
        f.read(4)  # magic
        num = struct.unpack('>I', f.read(4))[0]
        y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int8)

    return X, y


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch = Z.shape[0]
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z),axes=(1,)))
    correct = ndl.summation(Z * y_one_hot,axes=(1,))
    return ndl.summation(log_sum_exp-correct)/batch


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    n = X.shape[0]
    for i in range(0, n, batch):
        end = min(i + batch, n)
        X_batch = ndl.Tensor(X[i:end])

        y_one_hot = np.zeros((end - i, W2.shape[1]), dtype=np.float32)
        y_one_hot[np.arange(end - i), y[i:end]] = 1
        y_one_hot = ndl.Tensor(y_one_hot)

        logits = ndl.relu(X_batch @ W1) @ W2
        loss = softmax_loss(logits, y_one_hot)

        loss.backward()

        W1.data = (W1 - lr * W1.grad).data
        W2.data = (W2 - lr * W2.grad).data

    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
