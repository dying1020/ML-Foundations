import matplotlib.pyplot as plt
import numpy as np


def PLA(X, y):
    W = np.zeros(X.shape[1]) # W = 0
    update_cnt = 0
    done = False

    while not done:
        done = True
        for i in range(len(X)):
            sign = np.sign(np.dot(W, X[i]))
            if sign == 0:
                sign = -1

            if sign != y[i]:
                W = W + y[i] * X[i]
                update_cnt = update_cnt + 1
                done = False
    return W, update_cnt


def main():
    raw_data = np.loadtxt('hw1_7_train.dat')

    X = raw_data[:, :4]
    y = raw_data[:, -1]
    print('shape of X:', X.shape)
    print('shape of y:', y.shape)

    # add $x_0 = 1$ to each $x_n$
    X = np.insert(X, 0, 1, axis=1)
    print('shape of X:', X.shape)

    total_update_cnt = 0
    history = []
    for i in range(1126):
        s = np.arange(X.shape[0])
        np.random.shuffle(s)
        X_rand = X[s]
        y_rand = y[s]
        W, update_cnt = PLA(X_rand, y_rand)
        total_update_cnt = total_update_cnt + update_cnt
        history.append(update_cnt)
    print('Average number of updates:', total_update_cnt/1126)

    plt.hist(history, bins='auto')
    plt.title('Histogram of PLA')
    plt.xlabel('# of updates')
    plt.ylabel('frequency')
    plt.show()


if __name__ == "__main__":
    main()
