import matplotlib.pyplot as plt
import numpy as np


def gen_data(data_size):
    X = np.random.uniform(low=-1, high=1, size=data_size)
    Y = np.sign(X)
    # Add noise with 20% probability
    noise = np.random.uniform(size=data_size)
    noise = [-1 if i else 1 for i in (noise > 0.8)]
    Y = Y * noise
    # Sort by x value
    sort_index = X.argsort()
    X = X[sort_index]
    Y = Y[sort_index]
    return X, Y


def h(x, theta, s):
    return s * np.sign(x-theta)


def E_out(theta, s):
    return 0.5 + 0.3 * s * (np.abs(theta)-1)


def E_in(X, Y, theta, s):
    err_cnt = 0
    for i in range(len(X)):
        if h(X[i], theta, s) != Y[i]:
            err_cnt += 1
    return err_cnt / len(X)


def decision_stump(X, Y):
    min_E_in = np.inf
    best_theta, best_s = 0, 0
    for i in range(len(X)-1):
        for s in [-1, 1]:
            theta = np.median(X[i:i+2])
            err = E_in(X, Y, theta, s)
            if err < min_E_in:
                best_theta, best_s = theta, s
                min_E_in = err

    # All +1
    theta, s = np.median([-1, X[0]]), 1
    err = E_in(X, Y, theta, s)
    if err < min_E_in:
        best_theta, best_s = theta, s
        min_E_in = err
    # All -1
    theta, s = np.median([X[-1], 1]), 1
    err = E_in(X, Y, theta, s)
    if err < min_E_in:
        best_theta, best_s = theta, s
        min_E_in = err

    return best_theta, best_s, min_E_in


def main():
    data_size = 20
    iter_time = 1000
    history = []

    for i in range(iter_time):
        X, Y = gen_data(data_size)
        theta, s, min_E_in = decision_stump(X, Y)
        diff = min_E_in - E_out(theta, s)
        history.append(diff)

    plt.hist(history, bins=np.linspace(-1, 1, 100))
    plt.show()


if __name__ == "__main__":
    main()
