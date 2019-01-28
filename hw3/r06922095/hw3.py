import numpy as np
import matplotlib.pyplot as plt


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def gradient(X, Y, w):
    grad = np.zeros(x_dim)
    for i in range(len(X)):
        grad += sigmoid(-Y[i]*np.dot(w, X[i])) * (-Y[i]*X[i])
    grad /= len(X)
    return grad


def logistic_regression(X, Y, eta, T):
    # Initalize w, shape = dim of x
    w = np.zeros(x_dim)
    for t in range(T):
        grad = gradient(X, Y, w)
        w -= eta * grad
    return w


def stochastic_gradient(X, Y, w, i):
    grad = sigmoid(-Y[i]*np.dot(w, X[i])) * (-Y[i]*X[i])
    return grad


def logistic_regression_sgd(X, Y, eta, T):
    # Initalize w, shape = dim of x
    w = np.zeros(x_dim)
    sample_index = 0
    for t in range(T):
        grad = stochastic_gradient(X, Y, w, sample_index)
        w -= eta * grad
        sample_index = (sample_index + 1) % len(X)
    return w


def eval_error(X, Y, w):
    Y_hat = np.sign(np.dot(X, w))
    return len(Y_hat[Y_hat != Y]) / len(X)


train_data = np.loadtxt('./hw3_train.dat.txt')
test_data = np.loadtxt('./hw3_test.dat.txt')

X_train, Y_train = train_data[:, :-1], train_data[:, -1]
X_test, Y_test = test_data[:, :-1], test_data[:, -1]
x_dim = X_train.shape[1]

T = 2000

# Question 4

# Gradient descent
E_in_gd = []
w = np.zeros(x_dim)
for t in range(T):
    grad = gradient(X_train, Y_train, w)
    w -= 0.01 * grad
    E_in_gd.append(eval_error(X_train, Y_train, w))

# Stochastic Gradient descent
E_in_sgd = []
w = np.zeros(x_dim)
sample_index = 0
for t in range(T):
    grad = stochastic_gradient(X_train, Y_train, w, sample_index)
    w -= 0.001 * grad
    sample_index = (sample_index + 1) % len(X_train)
    E_in_sgd.append(eval_error(X_train, Y_train, w))

plt.plot(E_in_gd, label='E_in gd')
plt.plot(E_in_sgd, label='E_in sgd')
plt.xlabel('t: iteration')
plt.ylabel('in-sample error')
plt.legend()
plt.show()

# Question 5

# Gradient descent
E_out_gd = []
w = np.zeros(x_dim)
for t in range(T):
    grad = gradient(X_train, Y_train, w)
    w -= 0.01 * grad
    E_out_gd.append(eval_error(X_test, Y_test, w))

# Stochastic gradient descent
E_out_sgd = []
w = np.zeros(x_dim)
sample_index = 0
for t in range(T):
    grad = stochastic_gradient(X_train, Y_train, w, sample_index)
    w -= 0.001 * grad
    sample_index = (sample_index + 1) % len(X_train)
    E_out_sgd.append(eval_error(X_test, Y_test, w))

plt.plot(E_out_gd, label='E_out gd')
plt.plot(E_out_sgd, label='E_out sgd')
plt.xlabel('t: iteration')
plt.ylabel('out-sample error')
plt.legend()
plt.show()
