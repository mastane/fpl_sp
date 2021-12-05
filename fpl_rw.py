# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import comb_ucb1 as ucb1
from scipy.stats import bernoulli
import time
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import beta


def opt(s, l):
    return s[np.argmin(np.dot(s, l))]


def perturb_exp(t, dim, mk):
    return np.random.exponential(1, (t, mk+1, dim))


def perturb_gauss(t, dim, mk):
    return np.random.normal(0, 1, (t, mk+1, dim))


def perturb_gumbel(t, dim, mk):
    return np.random.gumbel(0, np.sqrt(6)/np.pi, (t, mk+1, dim))


def perturb_exp_2(t, dim, mk):
    z = np.zeros((t, mk+1, dim))
    for ti in range(t):
        for i in range(mk+1):
            z[ti][i] = perturb_vect_exp_2(dim)
    return z


def perturb_unif(t, dim, mk):
    return np.random.rand(t, mk+1, dim)


def perturb_vect_exp(dim):
    return np.random.exponential(1, dim)


def perturb_vect_gauss(dim):
    return np.random.normal(0, 1, dim)


def perturb_vect_gumbel(dim):
    return np.random.gumbel(0, np.sqrt(6)/np.pi, dim)


def wallis(n):  # using the beta function
    return 0.5*beta((n+1)/2, 1/2)


def sin_cdf(n, x):
    return 1/(2*wallis(n))*quad(lambda t: np.sin(t)**n, 0, x)[0]


def perturb_vect_exp_2(dim):  # Use Box Muller like method
    if dim == 1:
        u = np.random.exponential(2, size=1)
        if np.random.randint(2) == 0:
            return u
        return -u
    theta = np.zeros(dim-1)
    theta[dim-2] = 2*np.pi*np.random.rand()
    for i in range(dim-2):
        y = np.random.rand()
        theta[i] = fsolve(lambda s: sin_cdf(dim-2-i, s)-y, np.pi/2)[0]
    r = sum(np.random.normal(0, 1, 2*dim)**2)
    x = np.zeros(dim)
    x[0] = r*np.cos(theta[0])
    x[dim-1] = r*np.sin(theta[0])
    for i in range(1, dim-1):
        x[dim-1] *= np.sin(theta[i])
        x[i] = r*np.cos(theta[i])
        for j in range(i):
            x[i] *= np.sin(theta[j])
    return x/(2*np.sqrt(dim+1))  # because E(X1**2) = E(R**2)/dim and, by integrating, E(R**2) = 4*(dim+1)


def perturb_vect_unif(dim):
    return np.random.rand(dim)


def perturb(t, dim, mk, method="exp"):
    if method == "exp":
        return perturb_exp(t, dim, mk)
    elif method == "gauss":
        return perturb_gauss(t, dim, mk)
    elif method == "gumbel":
        return perturb_gumbel(t, dim, mk)
    elif method == "exp_2":
        return perturb_exp_2(t, dim, mk)


def perturb_vect(dim, method="exp"):
    if method == "exp":
        return perturb_vect_exp(dim)
    elif method == "gauss":
        return perturb_vect_gauss(dim)
    elif method == "gumbel":
        return perturb_vect_gumbel(dim)
    elif method == "exp_2":
        return perturb_vect_exp_2(dim)


def fpl_rw(eta, mk, l, s, method="exp", high=False, md=1):  # s is included in {0, 1}^d
    (t, dim) = l.shape
    c = np.zeros(t)
    l_est = np.zeros(dim)
    beta_high = beta_opt_high(dim, t, md)
    for ti in range(t):
        loss = l[ti]
        # compute perturbed leader
        z = perturb_vect(dim, method)
        perturbed_est = eta*l_est-z
        v = opt(s, perturbed_est)
        # suffer loss
        c[ti] = np.dot(v, loss)
        # compute recurrence weights
        k = rw(eta*l_est, v, mk, s, dim, method)
        # update loss estimates
        if high:
            l_est += 1/beta_high*np.log(1+beta_high*k*v*loss)
        else:
            l_est += k*v*loss
    return c


def rw(y, v, mk, s, dim, method, trunc=False, b=1):
    k = np.zeros(dim)
    # initialize waiting list
    waiting = np.array(v)
    if trunc:
        for i in range(mk):
            # increment counter
            k += waiting
            # compute perturbed leader
            z = -np.log(1-np.random.rand(dim)*(1-np.exp(-b)))
            perturbed_loss = y-z
            w = opt(s, perturbed_loss)
            # update waiting list
            waiting *= 1-v*w
            if sum([waiting[j] == 0 for j in range(dim)]) == dim:
                return k
        return k
    for i in range(mk):
        # increment counter
        k += waiting
        # compute perturbed leader
        z = perturb_vect(dim, method)
        perturbed_loss = y-z
        w = opt(s, perturbed_loss)
        # update waiting list
        waiting *= 1-v*w
        if sum([waiting[j] == 0 for j in range(dim)]) == dim:
            return k
    return k


def fpl_rw_adaptive(l, s, md, method="exp", trunc=False, high=False):  # with adaptive learning rates
    (t, dim) = l.shape
    c = np.zeros(t)
    l_est = np.zeros(dim)
    eta0 = np.log(dim/md) + 1
    si = 0
    beta_high = beta_opt_high(dim, t, md)
    if trunc:
        for ti in range(t):
            eta = np.sqrt(eta0/(1/eta0 + si))
            gamma = md*eta
            beta = (md/dim)*eta
            b = -np.log(beta)
            mk = np.random.geometric(min(1, gamma))
            loss = l[ti]
            # compute perturbed leader
            z = -np.log(1-np.random.rand(dim)*(1-np.exp(-b)))
            perturbed_est = eta*l_est-z
            v = opt(s, perturbed_est)
            # suffer loss
            c[ti] = np.dot(v, loss)
            # compute recurrence weights
            k = rw(eta*l_est, v, mk, s, dim, method=method, trunc=trunc, b=b)
            # update loss estimates
            if high:
                l_est += 1/beta_high*np.log(1+beta_high*k*v*loss)
                # update learning rate
                si += sum(1/beta_high*np.log(1+beta_high*k*v*loss))
            else:
                l_est += k*v*loss
                # update learning rate
                si += sum(k*v*loss)
    else:
        for ti in range(t):
            eta = np.sqrt(eta0/(1/eta0 + si))
            gamma = md*eta
            loss = l[ti]
            # compute perturbed leader
            z = perturb_vect(dim, method)
            perturbed_est = eta*l_est-z
            v = opt(s, perturbed_est)
            # suffer loss
            c[ti] = np.dot(v, loss)
            # compute recurrence weights
            mk = np.random.geometric(min(1, gamma))
            k = rw(eta*l_est, v, mk, s, dim, method=method)
            # update loss estimates
            if high:
                l_est += 1/beta_high*np.log(1+beta_high*k*v*loss)
                # update learning rate
                si += sum(1/beta_high*np.log(1+beta_high*k*v*loss))
            else:
                l_est += k*v*loss
                # update learning rate
                si += sum(k*v*loss)
    return c


def s_generator(dim, nb):
    return np.floor(np.random.rand(nb, dim) + 0.5)


def expected_regret(l, s, nit, method="exp", eta=1, mk=100, adapt=False, md=1, trunc=False, high=False):
    (t, dim) = l.shape
    l_sum = np.zeros(dim)
    loss_sum = 0
    if method != "comb_ucb1":
        for _ in range(nit):
            if adapt:
                loss_sum += np.sum(fpl_rw_adaptive(l, s, md, method=method, trunc=trunc, high=high))
            else:
                loss_sum += np.sum(fpl_rw(eta, mk, l, s, method, high=high, md=md))
        for ti in range(t):
            l_sum += l[ti]
    else:
        for _ in range(nit):
            cb_ucb1 = ucb1.comb_ucb1(l, s)
            t0 = cb_ucb1[1]
            loss_sum += np.sum(cb_ucb1[0][t0-1:])
        for ti in range(t-t0+1):
            l_sum += l[t0+ti-1]
    return loss_sum/nit - np.dot(opt(s, l_sum), l_sum)


def expected_regrets(l, s, nit, method="exp", eta=1, mk=100, adapt=False, md=1, trunc=False, high=False):
    (t, dim) = l.shape
    l_sum = np.zeros(dim)
    if method != "comb_ucb1":
        avgs = np.zeros(t)
        expect_regrets = np.zeros(t)
        m2 = np.zeros(t)
        for i in range(nit):
            if adapt:
                cum_losses = np.cumsum(fpl_rw_adaptive(l, s, md, method=method, trunc=trunc, high=False))
            else:
                cum_losses = np.cumsum(fpl_rw(eta, mk, l, s, method=method, high=high, md=md))
            delta = cum_losses - avgs
            avgs += delta/(i+1)
            m2 += delta*(cum_losses - avgs)
        if nit < 2:
            var = np.zeros(t)
        else:
            var = m2/(nit - 1)  # sample variance
        for ti in range(t):
            l_sum += l[ti]
            expect_regrets[ti] = avgs[ti] - np.dot(opt(s, l_sum), l_sum)  # we subtract best arm cost at time ti
    else:
        print "ERROR : you should use the regret_vector function because CombUCB1 is deterministic"
        """for i in range(nit):
            cb_ucb1 = ucb1.comb_ucb1(l, s)
            if i == 0:
                t0 = cb_ucb1[1]
                avgs = np.zeros(t-t0+1)
                expect_regrets = np.zeros(t-t0+1)
                m2 = np.zeros(t-t0+1)
            cum_losses = np.cumsum(cb_ucb1[0][t0-1:])
            delta = cum_losses - avgs
            avgs += delta/(i+1)
            m2 += delta*(cum_losses - avgs)
        if nit < 2:
            var = np.zeros(t-t0+1)
        else:
            var = m2/(nit - 1)  # sample variance
        for ti in range(t-t0+1):
            l_sum += l[t0+ti-1]
            expect_regrets[ti] = avgs[ti] - np.dot(opt(s, l_sum), l_sum)  # we subtract best arm cost at time ti"""
        return None
    h95 = 1.96*np.sqrt(var)/nit  # 95% confidence interval
    low = expect_regrets - h95
    high = expect_regrets + h95
    return expect_regrets, low, high


def regret_vector(l, s, losses):  # regret values for ti in range(t) with respect to the best arm at time ti
    (t, dim) = l.shape
    t2 = len(losses)
    t0 = t-t2+1
    regret_vect = np.zeros(t-t0+1)
    l_sum = np.zeros(dim)
    loss_vector = np.cumsum(losses)
    for ti in range(t-t0+1):
        l_sum += l[t0+ti-1]
        regret_vect[ti] = loss_vector[ti] - np.dot(opt(s, l_sum), l_sum)
    return regret_vect


def upper_bound(dim, t, md):
    return 3*md*np.sqrt(2*dim*t*(np.log(dim/md)+1))


def eta_opt(dim, t, md):
    return np.sqrt((np.log(dim/md)+1)/(2*dim*t))


def mk_opt(dim, t, md):
    return int(np.ceil(np.sqrt(dim*t)/(np.e*md*np.sqrt(np.log(dim/md)+1))))


def eta_opt_high(dim, t, md):
    return np.sqrt((np.log(dim/md)+1)/(dim*t))


def mk_opt_high(dim, t, md):
    return int(np.ceil(np.sqrt(dim*t/md)))


def beta_opt_high(dim, t, md):
    return np.sqrt(md/(dim*t))


if __name__ == "__main__":

    T = 1000
    d = 2
    # m = d
    m = 1
    # S = s_generator(d, n)
    S = np.eye(d)
    num = 10

    """L = np.ones((T, d))
    L[:, 0] = 0
    L[:, 2] = 0
    L[:, 4] = 0
    L[:, 6] = 0
    L[:, 13] = 0
    L[:, 20] = 0"""
    # L = np.random.rand(T, d)
    L = np.zeros((T, d))
    # L[0] = np.random.rand(d)
    for Ti in range(int(T/2)):
        L[2*Ti][0] = 1
        L[2*Ti+1][1] = 1
    L[0][0] = 0.5
    print L


    # sd = 0.02
    # sd2 = 0.01
    # sd3 = 0.005

    # inc = np.random.normal(0, sd, (T-1, d))

    # inc2 = np.zeros((T-1, d))
    # inc2[np.floor((T-1)/3):] = np.random.normal(0, sd2, (T-1-np.floor((T-1)/3), d))
    # inc2[np.floor((T-1)/5):] = bernoulli.rvs(0.4, size=(T-1-np.floor((T-1)/5), d))*2*0.02 - 0.02
    # inc += inc2
    # inc3 = np.zeros((T-1, d))
    # inc3[np.floor((T-1)*2/3):] = np.random.normal(0, sd3, (T-1-np.floor((T-1)*2/3), d))
    # inc3[np.floor((T-1)*2/5):] = bernoulli.rvs(0.5, size=(T-1-np.floor((T-1)*2/5), d))*2*0.02 - 0.02
    # inc += inc3
    '''for Ti in range(T-1):
        for Idx in range(d):
            L[Ti+1][Idx] = L[Ti][Idx] + inc[Ti][Idx]
            # L[Ti+1][Idx] = L[Ti][Idx] + np.random.normal(0, sd)
            if L[Ti+1][Idx] < 0:
                L[Ti+1][Idx] = 0
            elif L[Ti+1][Idx] > 1:
                L[Ti+1][Idx] = 1'''
    '''for Idx in range(d):
        L[1:, Idx] = bernoulli.rvs(L[0][Idx], size=T-1)'''
    '''L[1:, 0] = bernoulli.rvs(0.1, size=T-1)'''
    '''L = bernoulli.rvs(0.5, size=(T, d))
    Delta = 0.1
    L[:, 0] = bernoulli.rvs(0.5-Delta, size=T)'''
    '''L2 = np.zeros((T, d))
    L2[0] = L[0]
    for Ti in range(T-1):
        for Idx in range(d):
            L2[Ti+1][Idx] = L2[Ti][Idx] + inc[Ti][Idx]
    for Ti in range(1, T):
        for Idx in range(d):
            if L2[Ti][Idx] < 0:
                L2[Ti][Idx] = 0
            elif L2[Ti][Idx] > 1:
                L2[Ti][Idx] = 1
    L3 = np.zeros((T, d))
    L3[0] = L[0]
    for Ti in range(T-1):
        for Idx in range(d):
            L3[Ti+1][Idx] = L3[0][Idx] + inc[Ti][Idx]
            if L3[Ti+1][Idx] < 0:
                L3[Ti+1][Idx] = 0
            elif L3[Ti+1][Idx] > 1:
                L3[Ti+1][Idx] = 1'''

    eta_const = eta_opt(d, T, m)
    # eta2 = 10
    # print "eta2 = {}".format(eta2)
    M = mk_opt(d, T, m)
    # meth = "exp"

    print 'T = {}'.format(T)
    print 'd = {}'.format(d)
    print 'm = {}'.format(m)
    print 'upper bound : = {}'.format(upper_bound(d, T, m))
    print 'eta optimal = {}'.format(eta_const)
    print 'M optimal = {}'.format(M)
    print 'num = {}'.format(num)

    begin = time.time()
    regrets_exp = expected_regrets(L, S, num, "exp", eta_const, M)
    end = time.time()
    print '........... finished regrets_exp in {} seconds'.format(end-begin)
    """begin = time.time()
    regrets_exp_eta2 = expected_regrets(L, S, num, "exp", eta2, M)
    end = time.time()
    print '........... finished regrets_exp_eta2 in {} seconds'.format(end-begin)"""
    begin = time.time()
    regrets_gauss = expected_regrets(L, S, num, "gauss", eta_const, M)
    end = time.time()
    print '........... finished regrets_gauss in {} seconds'.format(end-begin)
    begin = time.time()
    regrets_gumbel = expected_regrets(L, S, num, "gumbel", eta_const, M)
    end = time.time()
    print '........... finished regrets_gumbel in {} seconds'.format(end-begin)
    begin = time.time()
    regrets_exp_2 = expected_regrets(L, S, num, "exp_2", eta_const, M)
    end = time.time()
    print '........... finished regrets_exp_2 in {} seconds'.format(end-begin)
    begin = time.time()
    regrets_exp_adapt = expected_regrets(L, S, num, "exp", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets_exp_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets_gauss_adapt = expected_regrets(L, S, num, "gauss", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets_gauss_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets_gumbel_adapt = expected_regrets(L, S, num, "gumbel", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets_gumbel_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets_exp_2_adapt = expected_regrets(L, S, num, "exp_2", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets_exp_2_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets_exp_adapt_trunc = expected_regrets(L, S, num, "exp", eta_const, M, adapt=True, md=m, trunc=True)
    end = time.time()
    print '........... finished regrets_exp_adapt_trunc in {} seconds'.format(end-begin)
    begin = time.time()
    results_ucb1 = ucb1.comb_ucb1(L, S)
    l_ucb1 = results_ucb1[0]
    T0 = results_ucb1[1]
    regrets_ucb1 = regret_vector(L, S, l_ucb1[T0-1:])
    end = time.time()
    print '........... finished CombUCB1 for L in {} seconds'.format(end-begin)
    '''
    begin = time.time()
    regrets2_exp = expected_regrets(L2, S, num, "exp", eta_const, M)
    end = time.time()
    print '........... finished regrets2_exp in {} seconds'.format(end-begin)
    begin = time.time()
    regrets2_gauss = expected_regrets(L2, S, num, "gauss", eta_const, M)
    end = time.time()
    print '........... finished regrets2_gauss in {} seconds'.format(end-begin)
    begin = time.time()
    regrets2_gumbel = expected_regrets(L2, S, num, "gumbel", eta_const, M)
    end = time.time()
    print '........... finished regrets2_gumbel in {} seconds'.format(end-begin)
    begin = time.time()
    regrets2_exp_2 = expected_regrets(L2, S, num, "exp_2", eta_const, M)
    end = time.time()
    print '........... finished regrets2_exp_2 in {} seconds'.format(end-begin)
    begin = time.time()
    regrets2_exp_adapt = expected_regrets(L2, S, num, "exp", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets2_exp_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets2_gauss_adapt = expected_regrets(L2, S, num, "gauss", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets2_gauss_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets2_gumbel_adapt = expected_regrets(L2, S, num, "gumbel", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets2_gumbel_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets2_exp_2_adapt = expected_regrets(L2, S, num, "exp_2", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets2_exp_2_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets2_exp_adapt_trunc = expected_regrets(L2, S, num, "exp", eta_const, M, adapt=True, md=m, trunc=True)
    end = time.time()
    print '........... finished regrets2_exp_adapt_trunc in {} seconds'.format(end-begin)
    begin = time.time()
    results_ucb12 = ucb1.comb_ucb1(L2, S)
    l_ucb12 = results_ucb12[0]
    T02 = results_ucb12[1]
    regrets_ucb12 = regret_vector(L2, S, l_ucb12[T02-1:])
    end = time.time()
    print '........... finished CombUCB1 for L2 in {} seconds'.format(end-begin)

    begin = time.time()
    regrets3_exp = expected_regrets(L3, S, num, "exp", eta_const, M)
    end = time.time()
    print '........... finished regrets3_exp in {} seconds'.format(end-begin)
    begin = time.time()
    regrets3_gauss = expected_regrets(L3, S, num, "gauss", eta_const, M)
    end = time.time()
    print '........... finished regrets3_gauss in {} seconds'.format(end-begin)
    begin = time.time()
    regrets3_gumbel = expected_regrets(L3, S, num, "gumbel", eta_const, M)
    end = time.time()
    print '........... finished regrets3_gumbel in {} seconds'.format(end-begin)
    begin = time.time()
    regrets3_exp_2 = expected_regrets(L3, S, num, "exp_2", eta_const, M)
    end = time.time()
    print '........... finished regrets3_exp_2 in {} seconds'.format(end-begin)
    begin = time.time()
    regrets3_exp_adapt = expected_regrets(L3, S, num, "exp", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets3_exp_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets3_gauss_adapt = expected_regrets(L3, S, num, "gauss", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets3_gauss_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets3_gumbel_adapt = expected_regrets(L3, S, num, "gumbel", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets3_gumbel_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets3_exp_2_adapt = expected_regrets(L3, S, num, "exp_2", eta_const, M, adapt=True, md=m)
    end = time.time()
    print '........... finished regrets3_exp_2_adapt in {} seconds'.format(end-begin)
    begin = time.time()
    regrets3_exp_adapt_trunc = expected_regrets(L3, S, num, "exp", eta_const, M, adapt=True, md=m, trunc=True)
    end = time.time()
    print '........... finished regrets3_exp_adapt_trunc in {} seconds'.format(end-begin)
    begin = time.time()
    results_ucb13 = ucb1.comb_ucb1(L3, S)
    l_ucb13 = results_ucb13[0]
    T03 = results_ucb13[1]
    regrets_ucb13 = regret_vector(L3, S, l_ucb13[T03-1:])
    end = time.time()
    print '........... finished CombUCB1 for L3 in {} seconds'.format(end-begin)
    '''
    t_vect = np.arange(T) + 1
    # t_vect_ucb1 = np.arange(T-T0+1) + T0 - 1

    '''plt.figure()
    plt.suptitle("FPL with RW in semi-bandit feedback - Multi armed bandit")
    plt.subplot(211)
    plt.plot(t_vect, np.cumsum(res))
    plt.title("Optimal learning rate")
    plt.xlabel("time")
    plt.ylabel("learner's cumulative loss")

    plt.subplot(212)
    plt.plot(t_vect, np.cumsum(res_adapt))
    plt.title("Adaptive learning rate")
    plt.xlabel("time")
    plt.ylabel("learner's cumulative loss")

    plt.figure()
    plt.plot(t_vect, np.cumsum(res_ucb1))
    plt.title("CombUCB1 algorithm")
    plt.xlabel("time")
    plt.ylabel("learner's cumulative loss")'''

    '''plt.figure()
    plt.title("FPL+RW")
    if meth != "comb_ucb1":
        plt.plot(t_vect, regrets[0])
        plt.fill_between(t_vect, regrets[1], regrets[2])
    else:
        plt.plot(t_vect_ucb1, regrets2)
    plt.xlabel("time")
    plt.ylabel("regret")'''

    phaal = 0.35

    # plt.figure()
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.title("FPL+RW vs CombUCB1 - Classical tricky losses for d = 2")
    # plt.title("FPL+RW vs CombUCB1 - Bernoulli(0.5) losses with one best path(0.5-Delta), Delta = {}, d = {}".format(Delta, d))
    # plt.title("FPL+RW vs CombUCB1 - Bernoulli losses, d = {}".format(d))
    # plt.title("FPL+RW vs CombUCB1 - Bouncing random walk losses, sd = {}, d = {}".format(sd, d))
    plt.plot(t_vect, regrets_exp[0], label='exp', color='red')
    plt.fill_between(t_vect, regrets_exp[1], regrets_exp[2], color='red', alpha=phaal)
    '''plt.plot(t_vect, regrets_exp_eta2[0], label='exp with eta2', color='black')
    plt.fill_between(t_vect, regrets_exp_eta2[1], regrets_exp_eta2[2], color='black')'''
    plt.plot(t_vect, regrets_gauss[0], label='gauss', color='magenta')
    plt.fill_between(t_vect, regrets_gauss[1], regrets_gauss[2], color='magenta', alpha=phaal)
    plt.plot(t_vect, regrets_gumbel[0], label='gumbel', color='green')
    plt.fill_between(t_vect, regrets_gumbel[1], regrets_gumbel[2], color='green', alpha=phaal)
    plt.plot(t_vect, regrets_exp_2[0], label='exp_2', color='purple')
    plt.fill_between(t_vect, regrets_exp_2[1], regrets_exp_2[2], color='purple', alpha=phaal)
    plt.plot(t_vect, regrets_exp_adapt[0], label='exp adaptive', color='yellow')
    plt.fill_between(t_vect, regrets_exp_adapt[1], regrets_exp_adapt[2], color='yellow', alpha=phaal)
    plt.plot(t_vect, regrets_gauss_adapt[0], label='gauss adaptive', color='grey')
    plt.fill_between(t_vect, regrets_gauss_adapt[1], regrets_gauss_adapt[2], color='grey', alpha=phaal)
    plt.plot(t_vect, regrets_gumbel_adapt[0], label='gumbel adaptive', color='blue')
    plt.fill_between(t_vect, regrets_gumbel_adapt[1], regrets_gumbel_adapt[2], color='blue', alpha=phaal)
    plt.plot(t_vect, regrets_exp_2_adapt[0], label='exp_2 adaptive', color='firebrick')
    plt.fill_between(t_vect, regrets_exp_2_adapt[1], regrets_exp_2_adapt[2], color='firebrick', alpha=phaal)
    plt.plot(t_vect, regrets_exp_adapt_trunc[0], label='exp adaptive truncated', color='cyan')
    plt.fill_between(t_vect, regrets_exp_adapt_trunc[1], regrets_exp_adapt_trunc[2], color='cyan', alpha=phaal)
    plt.plot(np.arange(T-T0+1)+T0-1, regrets_ucb1, label='CombUCB1', color='black')
    # plt.legend(loc='upper left')
    # Shrink current axis by 10%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("time")
    plt.ylabel("regret")

    plt.figure()
    plt.title("Classical tricky losses for d = 2")
    # plt.title("Bernoulli(0.5) losses with one best path(0.5-Delta), Delta = {}, d = {}".format(Delta, d))
    # plt.title("Bernoulli losses, d = {}".format(d))
    # plt.title("Bouncing random walk losses, sd = {}, d = {}".format(sd, d))
    plt.plot(t_vect, L)
    plt.xlabel("time")
    plt.ylabel("loss")
    '''
    # plt.figure()
    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    plt.title("FPL+RW vs CombUCB1 - Not bouncing random walk losses, sd = {}, d = {}".format(sd, d))
    plt.plot(t_vect, regrets2_exp[0], label='exp', color='red')
    plt.fill_between(t_vect, regrets2_exp[1], regrets2_exp[2], color='red', alpha=phaal)
    plt.plot(t_vect, regrets2_gauss[0], label='gauss', color='magenta')
    plt.fill_between(t_vect, regrets2_gauss[1], regrets2_gauss[2], color='magenta', alpha=phaal)
    plt.plot(t_vect, regrets2_gumbel[0], label='gumbel', color='green')
    plt.fill_between(t_vect, regrets2_gumbel[1], regrets2_gumbel[2], color='green', alpha=phaal)
    plt.plot(t_vect, regrets2_exp_2[0], label='exp_2', color='purple')
    plt.fill_between(t_vect, regrets2_exp_2[1], regrets2_exp_2[2], color='purple', alpha=phaal)
    plt.plot(t_vect, regrets2_exp_adapt[0], label='exp adaptive', color='yellow')
    plt.fill_between(t_vect, regrets2_exp_adapt[1], regrets2_exp_adapt[2], color='yellow', alpha=phaal)
    plt.plot(t_vect, regrets2_gauss_adapt[0], label='gauss adaptive', color='grey')
    plt.fill_between(t_vect, regrets2_gauss_adapt[1], regrets2_gauss_adapt[2], color='grey', alpha=phaal)
    plt.plot(t_vect, regrets2_gumbel_adapt[0], label='gumbel adaptive', color='blue')
    plt.fill_between(t_vect, regrets2_gumbel_adapt[1], regrets2_gumbel_adapt[2], color='blue', alpha=phaal)
    plt.plot(t_vect, regrets3_exp_2_adapt[0], label='exp_2 adaptive', color='firebrick')
    plt.fill_between(t_vect, regrets3_exp_2_adapt[1], regrets3_exp_2_adapt[2], color='firebrick', alpha=phaal)
    plt.plot(t_vect, regrets2_exp_adapt_trunc[0], label='exp adaptive truncated', color='cyan')
    plt.fill_between(t_vect, regrets2_exp_adapt_trunc[1], regrets2_exp_adapt_trunc[2], color='cyan', alpha=phaal)
    plt.plot(np.arange(T-T02+1)+T02-1, regrets_ucb12, label='CombUCB1', color='black')
    # plt.legend(loc='upper left')
    # Shrink current axis by 10%
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("time")
    plt.ylabel("regret")

    plt.figure()
    plt.title("Not bouncing random walk losses, sd = {}, d = {}".format(sd, d))
    plt.plot(t_vect, L2)
    plt.xlabel("time")
    plt.ylabel("loss")

    # plt.figure()
    fig3 = plt.figure()
    ax3 = plt.subplot(111)
    plt.title("FPL+RW vs CombUCB1 - Normal losses, sd = {}, d = {}".format(sd, d))
    plt.plot(t_vect, regrets3_exp[0], label='exp', color='red')
    plt.fill_between(t_vect, regrets3_exp[1], regrets3_exp[2], color='red', alpha=phaal)
    plt.plot(t_vect, regrets3_gauss[0], label='gauss', color='magenta')
    plt.fill_between(t_vect, regrets3_gauss[1], regrets3_gauss[2], color='magenta', alpha=phaal)
    plt.plot(t_vect, regrets3_gumbel[0], label='gumbel', color='green')
    plt.fill_between(t_vect, regrets3_gumbel[1], regrets3_gumbel[2], color='green', alpha=phaal)
    plt.plot(t_vect, regrets3_exp_2[0], label='exp_2', color='purple')
    plt.fill_between(t_vect, regrets3_exp_2[1], regrets3_exp_2[2], color='purple', alpha=phaal)
    plt.plot(t_vect, regrets3_exp_adapt[0], label='exp adaptive', color='yellow')
    plt.fill_between(t_vect, regrets3_exp_adapt[1], regrets3_exp_adapt[2], color='yellow', alpha=phaal)
    plt.plot(t_vect, regrets3_gauss_adapt[0], label='gauss adaptive', color='grey')
    plt.fill_between(t_vect, regrets3_gauss_adapt[1], regrets3_gauss_adapt[2], color='grey', alpha=phaal)
    plt.plot(t_vect, regrets3_gumbel_adapt[0], label='gumbel adaptive', color='blue')
    plt.fill_between(t_vect, regrets3_gumbel_adapt[1], regrets3_gumbel_adapt[2], color='blue', alpha=phaal)
    plt.plot(t_vect, regrets3_exp_2_adapt[0], label='exp_2 adaptive', color='firebrick')
    plt.fill_between(t_vect, regrets3_exp_2_adapt[1], regrets3_exp_2_adapt[2], color='firebrick', alpha=phaal)
    plt.plot(t_vect, regrets3_exp_adapt_trunc[0], label='exp adaptive truncated', color='cyan')
    plt.fill_between(t_vect, regrets3_exp_adapt_trunc[1], regrets3_exp_adapt_trunc[2], color='cyan', alpha=phaal)
    plt.plot(np.arange(T-T03+1)+T03-1, regrets_ucb13, label='CombUCB1', color='black')
    # plt.legend(loc='upper left')
    # Shrink current axis by 10%
    box = ax.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("time")
    plt.ylabel("regret")

    plt.figure()
    plt.title("Normal losses, sd = {}, d = {}".format(sd, d))
    plt.plot(t_vect, L3)
    plt.xlabel("time")
    plt.ylabel("loss")
    '''
    """
    '''fig = plt.figure()
    ax = plt.subplot(111)
    plt.title("FPL+RW vs CombUCB1 - Bouncing random walk losses, sd = {}, d = {}".format(sd, d))
    plt.plot(t_vect, regrets_exp[0], label='exp', color='red')
    plt.plot(t_vect, np.sqrt(t_vect))
    plt.plot(t_vect, t_vect)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.fill_between(t_vect, regrets_exp[1], regrets_exp[2], color='magenta')
    plt.plot(t_vect, regrets_exp_eta2[0], label='exp with eta2', color='black')
    plt.fill_between(t_vect, regrets_exp_eta2[1], regrets_exp_eta2[2], color='grey')
    '''plt.plot(t_vect, regrets_gauss[0], label='gauss')
    plt.fill_between(t_vect, regrets_gauss[1], regrets_gauss[2])
    plt.plot(t_vect, regrets_gumbel[0], label='gumbel')
    plt.fill_between(t_vect, regrets_gumbel[1], regrets_gumbel[2])'''
    plt.plot(t_vect, regrets_exp_adapt[0], label='exp adaptive', color='green')
    plt.fill_between(t_vect, regrets_exp_adapt[1], regrets_exp_adapt[2], color='yellow')
    '''plt.plot(t_vect, regrets_gauss_adapt[0], label='gauss adaptive')
    plt.fill_between(t_vect, regrets_gauss_adapt[1], regrets_gauss_adapt[2])
    plt.plot(t_vect, regrets_gumbel_adapt[0], label='gumbel adaptive')
    plt.fill_between(t_vect, regrets_gumbel_adapt[1], regrets_gumbel_adapt[2])'''
    plt.plot(t_vect, regrets_exp_adapt_trunc[0], label='exp adaptive truncated')
    plt.fill_between(t_vect, regrets_exp_adapt_trunc[1], regrets_exp_adapt_trunc[2], color='blue')
    plt.plot(np.arange(T-T0+1)+T0-1, regrets_ucb1, label='CombUCB1', color='cyan')
    plt.legend(loc='upper left')
    plt.xlabel("time")
    plt.ylabel("regret")'''
    """
    plt.show()