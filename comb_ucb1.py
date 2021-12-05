# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import fpl_rw as fpl


def comb_ucb1(l, s):
    (t, dim) = l.shape
    c = np.zeros(t)
    u = np.zeros(dim)
    # initialization
    w1, t0 = init(l, s)
    w = np.zeros((t-t0+2, dim))
    w[0] = w1  # shift of 1 between here and notations of the article
    counter = np.zeros((2, dim))  # we just need to store counters for times ti-1 (counter[0]) and ti (counter[1])
    counter[0] = np.ones(dim)
    for ti in range(t-t0+1):
        loss = l[t0+ti-1]
        # compute UCBs
        for i in range(dim):
            u[i] = w[counter[0][i]-1][i] - radius(t0+ti-1, counter[0][i])  # lower bound of the confidence interval
        # solve the optimization problem
        a = fpl.opt(s, u)
        c[t0+ti-1] = np.dot(a, loss)
        # update statistics
        counter[1] = counter[0]
        for i in range(dim):
            if a[i] == 1:
                counter[1][i] += 1
                w[counter[1][i]-1][i] = (counter[0][i]*w[counter[0][i]-1][i] + loss[i])/counter[1][i]
        counter[0] = counter[1]
    return c, t0


def init(l, s):
    dim = len(s[0])
    w = np.zeros(dim)
    u = np.ones(dim)
    t = 1
    while 1 in u:
        a = fpl.opt(s, -u)  # negative sign here because we minimize
        for i in range(dim):
            if a[i] == 1:
                w[i] = l[t-1][i]
                u[i] = 0
        t += 1
    return w, t


def radius(x, y):
    return np.sqrt(1.5*np.log(x)/y)


if __name__ == "__main__":

    T = 3000
    d = 10
    # m = d
    m = 1
    # S = s_generator(d, n)
    S = np.eye(d)
    L = 1/2*np.ones((T, d))
    L[:, 0] = 0
    L[100:, 0] = 1/4
    # L = np.zeros((T, d))
    L = np.random.rand(T, d)
    num = 1

    results = comb_ucb1(L, S)
    res = results[0]
    T0 = results[1]

    regrets = fpl.expected_regrets(L, S, num, method="comb_ucb1")

    print 'd = {}'.format(d)
    print 'm = {}'.format(m)

    t_vect = np.arange(T-T0+1) + T0 - 1

    plt.figure()
    plt.plot(t_vect, np.cumsum(res[T0-1:]))
    plt.title("CombUCB1 algorithm")
    plt.xlabel("time")
    plt.ylabel("learner's cumulative loss")

    plt.figure()
    plt.title("CombUCB1 algorithm")
    plt.plot(t_vect, regrets[0])
    # plt.fill_between(t_vect, regrets[1], regrets[2], color='grey')
    plt.xlabel("time")
    plt.ylabel("regret")

    plt.show()