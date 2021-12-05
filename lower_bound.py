# -*- coding: utf-8 -*-

from __future__ import division
from fpl_rw import *
from scipy.stats import bernoulli
import shortest_path as shp


def xf(z, nb, t):
    x = np.random.randint(0, 2, (nb, t))
    x[z] = bernoulli.rvs(1/2-eps, size=t)
    return x


def expected_regret_lower_bound(s, nb, t, nit, eta, mk, method="exp"):
    z = np.random.randint(0, nb, nit)
    loss_sum = 0
    min_sum = 0
    for i in range(nit):
        l_sum = np.zeros(nb)
        x = xf(z[i], nb, t)
        l = np.transpose(x)
        loss_sum += sum(fpl_rw(eta, mk, l, s, method))
        for j in range(nb):
            l_sum[j] += sum(x[j])
        min_sum += np.dot(opt(s, l_sum), l_sum)
    return (loss_sum - min_sum)/nit


def expected_regret_lower_bound_graph(grid_size, nb, t, nit, method="exp", eta=1, mk=100):
    g = shp.grid(grid_size)
    z = np.random.randint(0, nb, nit)
    loss_sum = 0
    min_sum = 0
    loss_min = 0
    if method != "comb_ucb1":
        for i in range(nit):
            l_sum = np.zeros(nb)
            x = xf(z[i], nb, t)
            l = np.transpose(x)
            loss_sum += sum(shp.fpl_rw_graph(eta, mk, l, grid_size, method))
            for ti in range(t):
                l_sum += l[ti]
            for j, e in enumerate(g.edges()):
                g.edge[e[0]][e[1]]['weight'] = l_sum[j]
            path = shp.nx.shortest_path(g, 0, grid_size**2-1, 'weight')
            path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
            indices = [g.edges().index(e) for e in path_edges]
            for j in indices:
                loss_min += l_sum[j]
            min_sum += loss_min
            loss_min = 0
    else:
        for i in range(nit):
            l_sum = np.zeros(nb)
            x = xf(z[i], nb, t)
            l = np.transpose(x)
            cb_ucb1 = shp.comb_ucb1_graph(l, grid_size)
            t0 = cb_ucb1[1]
            loss_sum += np.sum(cb_ucb1[0][t0-1:])
            for ti in range(t-t0+1):
                l_sum += l[t0+ti-1]
            for j, e in enumerate(g.edges()):
                g.edge[e[0]][e[1]]['weight'] = l_sum[j]
            path = shp.nx.shortest_path(g, 0, grid_size**2-1, 'weight')
            path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
            indices = [g.edges().index(e) for e in path_edges]
            for j in indices:
                loss_min += l_sum[j]
            min_sum += loss_min
            loss_min = 0
    return (loss_sum - min_sum)/nit


def lower_bound(nb, t):
    return np.sqrt(t*nb)*(np.sqrt(2) - 1)/np.sqrt(32*np.log(4/3))


if __name__ == "__main__":

    T = 200
    d = 50
    S = np.eye(d)
    m = 1
    eta_const = eta_opt(d, T, m)
    M = mk_opt(d, T, m)
    n = d  # multi armed bandit case
    num = 50
    
    c = 1/(8*np.log(4/3))
    eps = np.sqrt(c*n/T)
    Z = np.random.randint(n)
    X = xf(Z, n, T)
    L = np.transpose(X)

    meth = "exp"
    M_big = 100
    Z_exp = perturb_exp(T, d, M_big)
    res = fpl_rw(eta_const, M, L, S, meth)
    res_adapt = fpl_rw_adaptive(L, S, m, meth)
    regret = expected_regret_lower_bound(S, n, T, num, eta_const, M, meth)
    
    print 'eta = {}'.format(eta_const)
    print 'M = {}'.format(M)
    print 'n = d = {}'.format(n)
    print 'm = {}'.format(m)
    print 'T = {}'.format(T)
    print 'eps = {}'.format(eps)
    print '...............expected regret = {}'.format(regret)
    print '...............theoretical upper bound : = {}'.format(upper_bound(d, T, m))
    print '...............theoretical lower bound : = {}'.format(lower_bound(n, T))
    
    t_vect = np.arange(T) + 1
    plt.figure()
    plt.suptitle("FPL with RW in semi-bandit feedback - Lower bounded case")
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
    
    plt.show()