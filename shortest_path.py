# -*- coding: utf-8 -*-

from __future__ import division
from fpl_rw import *
import networkx as nx
import comb_ucb1 as ucb1
from scipy.stats import bernoulli
import time
import itertools


def fpl_rw_graph(eta, mk, l, grid_size, method="exp", high=False, md=1, bary=False, basis=None):
    g = grid(grid_size)
    (t, dim) = l.shape
    cc = np.zeros(t)
    l_est = np.zeros(dim)
    beta_high = beta_opt_high(dim, t, md)
    if bary:
        dim2 = len(basis[0])
    for ti in range(t):
        loss = l[ti]
        # compute perturbed leader
        if bary:
            z = perturb_vect(dim2, method)
            z = np.dot(basis, z)
        else:
            z = perturb_vect(dim, method)
        perturbed_est = eta*l_est-z
        perturbed_est -= min(perturbed_est)  # BEWARE !!! Only equivalent if all paths have same length
        for i, e in enumerate(g.edges()):
            g.edge[e[0]][e[1]]['weight'] = perturbed_est[i]
        path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
        # suffer loss
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        indices = [g.edges().index(e) for e in path_edges]
        for i in indices:
            cc[ti] += loss[i]
        # compute recurrence weights
        k = rw_graph(eta*l_est, path_edges, mk, g, grid_size, dim, method, bary=bary, basis=basis)
        # update loss estimates
        if high:
            for e in path_edges:
                idx = g.edges().index(e)
                l_est[idx] += 1/beta_high*np.log(1+beta_high*k[e]*loss[idx])
        else:
            for e in path_edges:
                idx = g.edges().index(e)
                l_est[idx] += k[e]*loss[idx]
    return cc


def fpl_rw_adaptive_graph(l, grid_size, md, method="exp", trunc=False, high=False, bary=False, basis=None):
    g = grid(grid_size)
    (t, dim) = l.shape
    cc = np.zeros(t)
    l_est = np.zeros(dim)
    eta0 = np.log(dim/md) + 1
    si = 0
    beta_high = beta_opt_high(dim, t, md)
    if bary:
        dim2 = len(basis[0])
    if trunc:
        for ti in range(t):
            eta = np.sqrt(eta0/(1/eta0 + si))
            gamma = md*eta
            beta = (md/dim)*eta
            b = -np.log(beta)
            mk = np.random.geometric(min(1, gamma))
            loss = l[ti]
            # compute perturbed leader
            if bary:
                z = -np.log(1-np.random.rand(dim2)*(1-np.exp(-b)))
                z = np.dot(basis, z)
            else:
                z = -np.log(1-np.random.rand(dim)*(1-np.exp(-b)))
            perturbed_est = eta*l_est-z
            perturbed_est -= min(perturbed_est)
            for i, e in enumerate(g.edges()):
                g.edge[e[0]][e[1]]['weight'] = perturbed_est[i]
            path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
            # suffer loss
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            indices = [g.edges().index(e) for e in path_edges]
            for i in indices:
                cc[ti] += loss[i]
            # compute recurrence weights
            k = rw_graph(eta*l_est, path_edges, mk, g, grid_size, dim, trunc=trunc, b=b, bary=bary, basis=basis)
            # update loss estimates & si
            if high:
                for e in path_edges:
                    idx = g.edges().index(e)
                    update = 1/beta_high*np.log(1+beta_high*k[e]*loss[idx])
                    l_est[idx] += update
                    si += update
            else:
                for e in path_edges:
                    idx = g.edges().index(e)
                    update = k[e]*loss[idx]
                    l_est[idx] += update
                    si += update
    else:
        for ti in range(t):
            eta = np.sqrt(eta0/(1/eta0 + si))
            gamma = md*eta
            mk = np.random.geometric(min(1, gamma))
            loss = l[ti]
            # compute perturbed leader
            if bary:
                z = perturb_vect(dim2, method)
                z = np.dot(basis, z)
            else:
                z = perturb_vect(dim, method)
            perturbed_est = eta*l_est-z
            perturbed_est -= min(perturbed_est)
            for i, e in enumerate(g.edges()):
                g.edge[e[0]][e[1]]['weight'] = perturbed_est[i]
            path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
            # suffer loss
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            indices = [g.edges().index(e) for e in path_edges]
            for i in indices:
                cc[ti] += loss[i]
            # compute recurrence weights
            k = rw_graph(eta*l_est, path_edges, mk, g, grid_size, dim, method, bary=bary, basis=basis)
            # update loss estimates & si
            if high:
                for e in path_edges:
                    idx = g.edges().index(e)
                    update = 1/beta_high*np.log(1+beta_high*k[e]*loss[idx])
                    l_est[idx] += update
                    si += update
            else:
                for e in path_edges:
                    idx = g.edges().index(e)
                    update = k[e]*loss[idx]
                    l_est[idx] += update
                    si += update
    return cc


def rw_graph(l, p, mk, g, grid_size, dim, method="exp", trunc=False, b=1, bary=False, basis=None):
    k = {}
    if bary:
        dim2 = len(basis[0])
    for e in p:
        k[e] = 0
    # initialize waiting list
    waiting = list(p)
    if trunc:
        for _ in range(mk):
            # increment counter
            for e in waiting:
                k[e] += 1
            # compute perturbed leader
            if bary:
                z = -np.log(1-np.random.rand(dim2)*(1-np.exp(-b)))
                z = np.dot(basis, z)
            else:
                z = -np.log(1-np.random.rand(dim)*(1-np.exp(-b)))
            perturbed_loss = l-z
            perturbed_loss -= min(perturbed_loss)
            for j, e in enumerate(g.edges()):
                g.edge[e[0]][e[1]]['weight'] = perturbed_loss[j]
            path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            # update waiting list
            for e in list(set(path_edges) & set(waiting)):
                waiting.remove(e)
            if len(waiting) == 0:
                return k
        return k
    for _ in range(mk):
        # increment counter
        for e in waiting:
            k[e] += 1
        # compute perturbed leader
        if bary:
            z = perturb_vect(dim2, method)
            z = np.dot(basis, z)
        else:
            z = perturb_vect(dim, method)
        perturbed_loss = l-z
        perturbed_loss -= min(perturbed_loss)
        for j, e in enumerate(g.edges()):
            g.edge[e[0]][e[1]]['weight'] = perturbed_loss[j]
        path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        # update waiting list
        for e in list(set(path_edges) & set(waiting)):
            waiting.remove(e)
        if len(waiting) == 0:
            return k
    return k


def comb_ucb1_graph(l, grid_size):
    (t, dim) = l.shape
    g = grid(grid_size)
    c = np.zeros(t)
    u = np.zeros(dim)
    # initialization
    w1, t0 = init_graph(l, grid_size)
    w = np.zeros((t-t0+2, dim))
    w[0] = w1  # shift of 1 between here and notations of the article
    counter = np.zeros((2, dim))  # we just need to store counters for times ti-1 (counter[0]) and ti (counter[1])
    counter[0] = np.ones(dim)
    for ti in range(t-t0+1):
        loss = l[t0+ti-1]
        # compute UCBs
        for i in range(dim):
            u[i] = w[counter[0][i]-1][i] - ucb1.radius(t0+ti-1, counter[0][i])
        u -= min(u)
        # solve the optimization problem
        for i, e in enumerate(g.edges()):
            g.edge[e[0]][e[1]]['weight'] = u[i]
        path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
        # suffer loss
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        indices = [g.edges().index(e) for e in path_edges]
        for i in indices:
            c[t0+ti-1] += loss[i]
        # update statistics
        counter[1] = counter[0]
        for i in indices:
            counter[1][i] += 1
            w[counter[1][i]-1][i] = (counter[0][i]*w[counter[0][i]-1][i] + loss[i])/counter[1][i]
        counter[0] = counter[1]
    return c, t0


def init_graph(l, grid_size):
    dim = len(l[0])
    g = grid(grid_size)
    w = np.zeros(dim)
    u = -np.ones(dim)  # negative sign here because we minimize
    t = 1
    while -1 in u:  # and negative sign here too
        u -= min(u)
        for i, e in enumerate(g.edges()):
            g.edge[e[0]][e[1]]['weight'] = u[i]
        path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        indices = [g.edges().index(e) for e in path_edges]
        for i in indices:
            w[i] = l[t-1][i]
            u[i] = 0
        t += 1
    return w, t


def grid(g_size):
    g = nx.DiGraph()
    for i in range(g_size-1):
        for j in range(g_size-1):
            g.add_edge(i*g_size+j, i*g_size+j+1, weight=0)
            g.add_edge(i*g_size+j, (i+1)*g_size+j, weight=0)
    for i in range(g_size-1):
        g.add_edge((g_size-1)*g_size+i, (g_size-1)*g_size+i+1, weight=0)  # top of the grid, from left to right
        g.add_edge(i*g_size+g_size-1, (i+1)*g_size+g_size-1, weight=0)  # right side of the grid, from bottom to top
    return g


def expected_regret_graph(l, grid_size, nit, method="exp", eta=1, mk=100, adapt=False, md=1, trunc=False, high=False, bary=False, basis=None):
    (t, dim) = l.shape
    g = grid(grid_size)
    l_sum = np.zeros(dim)
    loss_sum = 0
    loss_min = 0
    if method != "comb_ucb1":
        for _ in range(nit):
            if adapt:
                loss_sum += np.sum(fpl_rw_adaptive_graph(l, grid_size, md, method=method, trunc=trunc, high=high, bary=bary, basis=basis))
            else:
                loss_sum += np.sum(fpl_rw_graph(eta, mk, l, grid_size, method=method, high=high, md=md, bary=bary, basis=basis))
        for ti in range(t):
            l_sum += l[ti]
    else:
        for _ in range(nit):
            cb_ucb1 = comb_ucb1_graph(l, grid_size)
            t0 = cb_ucb1[1]
            loss_sum += np.sum(cb_ucb1[0][t0-1:])
        for ti in range(t-t0+1):
            l_sum += l[t0+ti-1]
    for i, e in enumerate(g.edges()):
            g.edge[e[0]][e[1]]['weight'] = l_sum[i]
    path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    indices = [g.edges().index(e) for e in path_edges]
    for i in indices:
        loss_min += l_sum[i]
    return loss_sum/nit - loss_min


def expected_regrets_graph(l, grid_size, nit, method="exp", eta=1, mk=100, adapt=False, md=1, trunc=False, high=False, bary=False, basis=None):
    (t, dim) = l.shape
    g = grid(grid_size)
    l_sum = np.zeros(dim)
    loss_min = 0
    if method != "comb_ucb1":
        avgs = np.zeros(t)
        expect_regrets = np.zeros(t)
        m2 = np.zeros(t)
        for i in range(nit):
            if adapt:
                cum_losses = np.cumsum(fpl_rw_adaptive_graph(l, grid_size, md, method=method, trunc=trunc, high=high, bary=bary, basis=basis))
            else:
                cum_losses = np.cumsum(fpl_rw_graph(eta, mk, l, grid_size, method=method, high=high, md=md, bary=bary, basis=basis))
            delta = cum_losses - avgs
            avgs += delta/(i+1)
            m2 += delta*(cum_losses - avgs)
        if nit < 2:
            var = np.zeros(t)
        else:
            var = m2/(nit - 1)  # sample variance
        for ti in range(t):
            l_sum += l[ti]
            for i, e in enumerate(g.edges()):
                g.edge[e[0]][e[1]]['weight'] = l_sum[i]
            path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            indices = [g.edges().index(e) for e in path_edges]
            for i in indices:
                loss_min += l_sum[i]
            expect_regrets[ti] = avgs[ti] - loss_min  # we subtract best arm cost at time ti
            loss_min = 0
    else:
        print "ERROR : you should use the regret_vector_graph function because CombUCB1 is deterministic"
        """for i in range(nit):
            cb_ucb1 = comb_ucb1_graph(l, grid_size)
            if i == 0:
                t0 = cb_ucb1[1]
                avgs = np.zeros(t-t0+1)
                expect_regrets = np.zeros(t-t0+1)
                m2 = np.zeros(t-t0+1)
            losses = cb_ucb1[0][t0-1:]
            cum_losses = np.cumsum(losses)
            delta = cum_losses - avgs
            avgs += delta/(i+1)
            m2 += delta*(cum_losses - avgs)
        if nit < 2:
            var = np.zeros(t-t0+1)
        else:
            var = m2/(nit - 1)  # sample variance
        for ti in range(t-t0+1):
            l_sum += l[t0+ti-1]
            for i, e in enumerate(g.edges()):
                g.edge[e[0]][e[1]]['weight'] = l_sum[i]
            path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            indices = [g.edges().index(e) for e in path_edges]
            for i in indices:
                loss_min += l_sum[i]
            expect_regrets[ti] = avgs[ti] - loss_min  # we subtract best arm cost at time ti
            loss_min = 0"""
        return None
    h95 = 1.96*np.sqrt(var)/nit  # 95% confidence interval
    low = expect_regrets - h95
    high = expect_regrets + h95
    return expect_regrets, low, high


def regret_vector_graph(l, grid_size, losses):
    (t, dim) = l.shape
    t2 = len(losses)
    t0 = t-t2+1
    g = grid(grid_size)
    dim = len(l[0])
    regret_vect = np.zeros(t-t0+1)
    l_sum = np.zeros(dim)
    loss_vector = np.cumsum(losses)
    loss_min = 0
    for ti in range(t-t0+1):
        l_sum += l[t0+ti-1]
        for i, e in enumerate(g.edges()):
            g.edge[e[0]][e[1]]['weight'] = l_sum[i]
        path = nx.shortest_path(g, 0, grid_size**2-1, 'weight')
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        indices = [g.edges().index(e) for e in path_edges]
        for i in indices:
            loss_min += l_sum[i]
        regret_vect[ti] = loss_vector[ti] - loss_min
        loss_min = 0
    return regret_vect


'''def bary_spanner(g, c=2):
    grid_size = int(np.sqrt(len(g.nodes())))
    paths_gen = nx.all_simple_paths(g, source=0, target=grid_size**2-1)
    basis = np.eye(grid_size-1)
    paths_edges_list = []
    for p in paths_gen:
        path_edges = [(p[i], p[i+1]) for i in range(len(p)-1)]
        paths_edges_list.append(path_edges)
    for i in range(grid_size-1):
        b = np.array(basis)
        maxi = -1
        for p in paths_edges_list:
            b[i] = path_converter(p, grid_size)
            deter = np.abs(np.linalg.det(b))
            if deter > maxi:
                maxi = deter
                xi = np.array(b[i])
        basis[i] = xi
    finished = False
    while not finished:
        finished = True
        i = 0
        while finished and i < grid_size-1:
            b = np.array(basis)
            for p in paths_edges_list:
                b[i] = path_converter(p, grid_size)
                if np.abs(np.linalg.det(b)) > c*np.abs(np.linalg.det(basis)):
                    basis[i] = b[i]
                    finished = False
                    break
            i += 1
    return basis'''


def bary_spanner(g, c=2):
    grid_size = int(np.sqrt(len(g.nodes())))
    dim = 2*grid_size*(grid_size-1)
    basis = np.eye(dim)
    paths_edges = []
    paths_gen = nx.all_simple_paths(g, source=0, target=grid_size**2-1)
    cpt = 0
    idx = []
    for p in paths_gen:
        paths_edges.append(list(np.zeros(dim)))
        path_edges = [(p[i], p[i+1]) for i in range(len(p)-1)]
        indices = [g.edges().index(e) for e in path_edges]
        for i in indices:
            paths_edges[-1][i] = 1
    i = 0
    k = 0
    while i < dim - k:
        b = np.array(basis)
        maxi = -1
        print '......... k = ', k
        for p in paths_edges:
            pp = np.delete(p, idx)
            b[i] = pp
            deter = np.abs(np.linalg.det(b))
            if deter > maxi:
                maxi = deter
                xi = np.array(b[i])
        bool = False
        if maxi == 0:
            xi = np.zeros(dim)
            xi[i] = 1
            bool = True
            cpt += 1
            basis = np.delete(basis, i, 1)
            basis = np.delete(basis, i, 0)
            idx.append(i+len(idx))  # "+len(idx)" because of the shift caused by erasing previous vectors
        else:
            basis[i] = xi
            i += 1
        print 'i = ', i, ', e = ', g.edges()[i]
        print maxi, '@'
        # print b[i], '@@@'
        if not bool:
            print 'ok'
        else:
            print 'erased i-th vector for i =', i
        k = len(idx)
    print 'cpt =', cpt
    dim2 = len(basis)
    paths_cut = []
    for p in paths_edges:
        pp = np.delete(p, idx)
        paths_cut.append(pp)
    finished = False
    while not finished:
        finished = True
        i = 0
        while finished and i < dim2:
            print 'i =', i
            b = np.array(basis)
            for p in paths_cut:
                b[i] = np.array(p)
                if np.abs(np.linalg.det(b)) > c*np.abs(np.linalg.det(basis)):
                    basis[i] = np.array(b[i])
                    finished = False
                    break
            i += 1
    basis = basis.transpose()
    print 'det(basis) =', np.linalg.det(basis)
    basis2 = np.zeros((dim, dim2))
    for i in range(dim2):
        for j, p in enumerate(paths_cut):
            if np.array_equal(basis[:, i], p):
                basis2[:, i] = paths_edges[j]
                break
    return basis2  # the columns of basis (basis[:, i]) are in paths_cut (but not basis[i] !!!)


def path_converter(p, grid_size):  # a path is equivalent to the times of "move right" : 1 <= t1 < ... < t(N-1) <= 2*(N-1)
    if grid_size < 3:
        print "ERROR, grid is too small : grid_size < 3"
        return None
    v = np.zeros(grid_size-1)
    i = 0
    for e in p:
        v[i:] = v[i:] + 1
        if i == grid_size - 1:
            break
        if e[1] == e[0] + 1:
            i += 1
    return v


if __name__ == "__main__":

    T = 1000
    N = 3
    num = 10
    d = 2*N*(N-1)
    m = 2*(N-1)
    """L = np.ones((T, d))
    L[:, 0] = 0
    L[:, 2] = 0
    L[:, 4] = 0
    L[:, 6] = 0
    L[:, 13] = 0
    L[:, 20] = 0"""
    # L = np.random.rand(T, d)
    # L = np.zeros((T, d))  # L[0] = np.random.rand(d)

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
    '''for Idx in [0, 3, 8, 11]:
        L[1:, Idx] = bernoulli.rvs(0.1, size=T-1)'''
    L = bernoulli.rvs(0.5, size=(T, d))
    Delta = 0.1
    for Idx in [0, 3, 8, 11]:  # [0, 3, 9, 16, 18, 20]:
        L[:, Idx] = bernoulli.rvs(0.5-Delta, size=T)
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

    # eta_const_high = eta_opt_high(d, T, m)
    eta_const_high = eta_const
    # M_high = mk_opt_high(d, T, m)
    M_high = M

    begin = time.time()
    G = grid(N)
    basis = bary_spanner(G)
    end = time.time()
    print "creation of the grid of size {} took : {} seconds ".format(N, end-begin)
    # nx.draw_networkx(G)
    for Idx, E in enumerate(G.edges()):
        print '{} ... {}'.format(E, Idx)
    print 'T = {}'.format(T)
    print 'N = {}'.format(N)
    print 'd = {}'.format(d)
    print 'm = {}'.format(m)
    print 'upper bound : = {}'.format(upper_bound(d, T, m))

    print 'eta optimal = {}'.format(eta_const)
    print 'M optimal = {}'.format(M)
    print 'num = {}'.format(num)

    bary = True
    eta_vect = [0.001, 0.01, 0.1, 1, 10, 100]

    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    plt.title("FPL+RW vs CombUCB1 - Bernoulli(0.5) losses with one best path(0.5-Delta), Delta = {}, grid size = {}, bary = False".format(Delta, N))
    plt.xlabel("time")
    plt.ylabel("regret")

    '''fig2 = plt.figure()
    ax2 = plt.subplot(111)
    plt.title("FPL+RW vs CombUCB1 - Bernoulli(0.5) losses with one best path(0.5-Delta), Delta = {}, grid size = {}, bary = True".format(Delta, N))
    plt.xlabel("time")
    plt.ylabel("regret")'''
    for etai in eta_vect:
        '''begin = time.time()
        regrets_exp_bary = expected_regrets_graph(L, N, num, "exp", etai, M, bary=True, basis=basis)
        end = time.time()
        print '........... finished regrets_exp_bary in {} seconds'.format(end-begin)'''
        begin = time.time()
        regrets_exp = expected_regrets_graph(L, N, num, "exp_2", etai, M, bary=False, basis=basis)
        end = time.time()
        print '........... finished regrets_exp in {} seconds'.format(end-begin)
        """begin = time.time()
        regrets_exp_eta2 = expected_regrets_graph(L, N, num, "exp", eta2, M)
        end = time.time()
        print '........... finished regrets_exp_eta2 in {} seconds'.format(end-begin)"""
        '''begin = time.time()
        regrets_gauss = expected_regrets_graph(L, N, num, "gauss", eta_const, M)
        end = time.time()
        print '........... finished regrets_gauss in {} seconds'.format(end-begin)
        begin = time.time()
        regrets_gumbel = expected_regrets_graph(L, N, num, "gumbel", eta_const, M)
        end = time.time()
        print '........... finished regrets_gumbel in {} seconds'.format(end-begin)'''
        '''begin = time.time()
        regrets_exp_2 = expected_regrets_graph(L, N, num, "exp_2", eta_const, M)
        end = time.time()
        print '........... finished regrets_exp_2 in {} seconds'.format(end-begin)'''
        '''begin = time.time()
        regrets_exp_high = expected_regrets_graph(L, N, num, "exp", eta_const_high, M_high, high=True, md=m, bary=bary, basis=basis)
        end = time.time()
        print '........... finished regrets_exp_high in {} seconds'.format(end-begin)'''
        '''begin = time.time()
        regrets_exp_2_high = expected_regrets_graph(L, N, num, "exp_2", eta_const_high, M_high, high=True, md=m)
        end = time.time()
        print '........... finished regrets_exp_2_high in {} seconds'.format(end-begin)'''
        '''begin = time.time()
        regrets_exp_adapt = expected_regrets_graph(L, N, num, "exp", eta_const, M, adapt=True, md=m, bary=bary, basis=basis)
        end = time.time()
        print '........... finished regrets_exp_adapt in {} seconds'.format(end-begin)'''
        '''begin = time.time()
        regrets_gauss_adapt = expected_regrets_graph(L, N, num, "gauss", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets_gauss_adapt in {} seconds'.format(end-begin)
        begin = time.time()
        regrets_gumbel_adapt = expected_regrets_graph(L, N, num, "gumbel", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets_gumbel_adapt in {} seconds'.format(end-begin)'''
        '''begin = time.time()
        regrets_exp_2_adapt = expected_regrets_graph(L, N, num, "exp_2", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets_exp_2_adapt in {} seconds'.format(end-begin)'''
        '''begin = time.time()
        regrets_exp_adapt_high = expected_regrets_graph(L, N, num, "exp", eta_const_high, M_high, adapt=True, md=m, high=True, bary=bary, basis=basis)
        end = time.time()
        print '........... finished regrets_exp_adapt_high in {} seconds'.format(end-begin)'''
        '''begin = time.time()
        regrets_exp_2_adapt_high = expected_regrets_graph(L, N, num, "exp_2", eta_const_high, M_high, adapt=True, md=m, high=True)
        end = time.time()
        print '........... finished regrets_exp_2_adapt_high in {} seconds'.format(end-begin)'''
        '''begin = time.time()
        regrets_exp_adapt_trunc = expected_regrets_graph(L, N, num, "exp", eta_const, M, adapt=True, md=m, trunc=True, bary=bary, basis=basis)
        end = time.time()
        print '........... finished regrets_exp_adapt_trunc in {} seconds'.format(end-begin)'''
        begin = time.time()
        results_ucb1 = comb_ucb1_graph(L, N)
        l_ucb1 = results_ucb1[0]
        T0 = results_ucb1[1]
        regrets_ucb1 = regret_vector_graph(L, N, l_ucb1[T0-1:])
        end = time.time()
        print '........... finished CombUCB1 for L in {} seconds'.format(end-begin)
        """
        begin = time.time()
        regrets2_exp = expected_regrets_graph(L2, N, num, "exp", eta_const, M)
        end = time.time()
        print '........... finished regrets2_exp in {} seconds'.format(end-begin)
        '''begin = time.time()
        regrets2_gauss = expected_regrets_graph(L2, N, num, "gauss", eta_const, M)
        end = time.time()
        print '........... finished regrets2_gauss in {} seconds'.format(end-begin)
        begin = time.time()
        regrets2_gumbel = expected_regrets_graph(L2, N, num, "gumbel", eta_const, M)
        end = time.time()
        print '........... finished regrets2_gumbel in {} seconds'.format(end-begin)'''
        begin = time.time()
        regrets2_exp_2 = expected_regrets_graph(L2, N, num, "exp_2", eta_const, M)
        end = time.time()
        print '........... finished regrets2_exp_2 in {} seconds'.format(end-begin)
        begin = time.time()
        regrets2_exp_high = expected_regrets_graph(L2, N, num, "exp", eta_const_high, M_high, high=True, md=m)
        end = time.time()
        print '........... finished regrets2_exp_high in {} seconds'.format(end-begin)
        begin = time.time()
        regrets2_exp_2_high = expected_regrets_graph(L2, N, num, "exp_2", eta_const_high, M_high, high=True, md=m)
        end = time.time()
        print '........... finished regrets2_exp_2_high in {} seconds'.format(end-begin)
        begin = time.time()
        regrets2_exp_adapt = expected_regrets_graph(L2, N, num, "exp", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets2_exp_adapt in {} seconds'.format(end-begin)
        '''begin = time.time()
        regrets2_gauss_adapt = expected_regrets_graph(L2, N, num, "gauss", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets2_gauss_adapt in {} seconds'.format(end-begin)
        begin = time.time()
        regrets2_gumbel_adapt = expected_regrets_graph(L2, N, num, "gumbel", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets2_gumbel_adapt in {} seconds'.format(end-begin)'''
        begin = time.time()
        regrets2_exp_2_adapt = expected_regrets_graph(L2, N, num, "exp_2", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets2_exp_2_adapt in {} seconds'.format(end-begin)
        begin = time.time()
        regrets2_exp_adapt_high = expected_regrets_graph(L2, N, num, "exp", eta_const_high, M_high, adapt=True, md=m, high=True)
        end = time.time()
        print '........... finished regrets2_exp_adapt_high in {} seconds'.format(end-begin)
        begin = time.time()
        regrets2_exp_2_adapt_high = expected_regrets_graph(L2, N, num, "exp_2", eta_const_high, M_high, adapt=True, md=m, high=True)
        end = time.time()
        print '........... finished regrets2_exp_2_adapt_high in {} seconds'.format(end-begin)
        begin = time.time()
        regrets2_exp_adapt_trunc = expected_regrets_graph(L2, N, num, "exp", eta_const, M, adapt=True, md=m, trunc=True)
        end = time.time()
        print '........... finished regrets2_exp_adapt_trunc in {} seconds'.format(end-begin)
        begin = time.time()
        results_ucb12 = comb_ucb1_graph(L2, N)
        l_ucb12 = results_ucb12[0]
        T02 = results_ucb12[1]
        regrets_ucb12 = regret_vector_graph(L2, N, l_ucb12[T02-1:])
        end = time.time()
        print '........... finished CombUCB1 for L2 in {} seconds'.format(end-begin)

        begin = time.time()
        regrets3_exp = expected_regrets_graph(L3, N, num, "exp", eta_const, M)
        end = time.time()
        print '........... finished regrets3_exp in {} seconds'.format(end-begin)
        '''begin = time.time()
        regrets3_gauss = expected_regrets_graph(L3, N, num, "gauss", eta_const, M)
        end = time.time()
        print '........... finished regrets3_gauss in {} seconds'.format(end-begin)
        begin = time.time()
        regrets3_gumbel = expected_regrets_graph(L3, N, num, "gumbel", eta_const, M)
        end = time.time()
        print '........... finished regrets3_gumbel in {} seconds'.format(end-begin)'''
        begin = time.time()
        regrets3_exp_2 = expected_regrets_graph(L3, N, num, "exp_2", eta_const, M)
        end = time.time()
        print '........... finished regrets3_exp_2 in {} seconds'.format(end-begin)
        begin = time.time()
        regrets3_exp_high = expected_regrets_graph(L3, N, num, "exp", eta_const_high, M_high, high=True, md=m)
        end = time.time()
        print '........... finished regrets3_exp_high in {} seconds'.format(end-begin)
        begin = time.time()
        regrets3_exp_2_high = expected_regrets_graph(L3, N, num, "exp_2", eta_const_high, M_high, high=True, md=m)
        end = time.time()
        print '........... finished regrets3_exp_2_high in {} seconds'.format(end-begin)
        begin = time.time()
        regrets3_exp_adapt = expected_regrets_graph(L3, N, num, "exp", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets3_exp_adapt in {} seconds'.format(end-begin)
        '''begin = time.time()
        regrets3_gauss_adapt = expected_regrets_graph(L3, N, num, "gauss", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets3_gauss_adapt in {} seconds'.format(end-begin)
        begin = time.time()
        regrets3_gumbel_adapt = expected_regrets_graph(L3, N, num, "gumbel", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets3_gumbel_adapt in {} seconds'.format(end-begin)'''
        begin = time.time()
        regrets3_exp_2_adapt = expected_regrets_graph(L3, N, num, "exp_2", eta_const, M, adapt=True, md=m)
        end = time.time()
        print '........... finished regrets3_exp_2_adapt in {} seconds'.format(end-begin)
        begin = time.time()
        regrets3_exp_adapt_high = expected_regrets_graph(L3, N, num, "exp", eta_const_high, M_high, adapt=True, md=m, high=True)
        end = time.time()
        print '........... finished regrets3_exp_adapt_high in {} seconds'.format(end-begin)
        begin = time.time()
        regrets3_exp_2_adapt_high = expected_regrets_graph(L3, N, num, "exp_2", eta_const_high, M_high, adapt=True, md=m, high=True)
        end = time.time()
        print '........... finished regrets3_exp_2_adapt_high in {} seconds'.format(end-begin)
        begin = time.time()
        regrets3_exp_adapt_trunc = expected_regrets_graph(L3, N, num, "exp", eta_const, M, adapt=True, md=m, trunc=True)
        end = time.time()
        print '........... finished regrets3_exp_adapt_trunc in {} seconds'.format(end-begin)
        begin = time.time()
        results_ucb13 = comb_ucb1_graph(L3, N)
        l_ucb13 = results_ucb13[0]
        T03 = results_ucb13[1]
        regrets_ucb13 = regret_vector_graph(L3, N, l_ucb13[T03-1:])
        end = time.time()
        print '........... finished CombUCB1 for L3 in {} seconds'.format(end-begin)
        """
        t_vect = np.arange(T) + 1
        # t_vect_ucb1 = np.arange(T-T0+1) + T0 - 1

        '''plt.figure()
        plt.suptitle("FPL with RW in semi-bandit feedback - Shortest path")
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
        plt.title("FPL+RW - graph")
        if meth != "comb_ucb1":
            plt.plot(t_vect, regrets[0])
            plt.fill_between(t_vect, regrets[1], regrets[2])
        else:
            plt.plot(t_vect_ucb1, regrets2)
        plt.xlabel("time")
        plt.ylabel("regret")'''

        phaal = 0.35

        # plt.figure()
        '''fig = plt.figure()
        ax = plt.subplot(111)
        plt.title("FPL+RW vs CombUCB1 - Bernoulli(0.5) losses with one best path(0.5-Delta), Delta = {}, grid size = {}, eta = {}".format(Delta, N, etai))'''
        # plt.title("FPL+RW vs CombUCB1 - Bernoulli losses, grid size = {}".format(N))
        # plt.title("FPL+RW vs CombUCB1 - Bouncing random walk losses, sd = {}, grid size = {}".format(sd, N))
        ax1.plot(t_vect, regrets_exp[0], label='exp_2, eta={}'.format(etai))
        ax1.fill_between(t_vect, regrets_exp[1], regrets_exp[2], alpha=phaal)
        '''ax2.plot(t_vect, regrets_exp_bary[0], label='exp, bary=True, eta={}'.format(etai))
        ax2.fill_between(t_vect, regrets_exp_bary[1], regrets_exp_bary[2], alpha=phaal)'''
        '''plt.plot(t_vect, regrets_exp_eta2[0], label='exp with eta2', color='black')
        plt.fill_between(t_vect, regrets_exp_eta2[1], regrets_exp_eta2[2], color='black')'''
        '''plt.plot(t_vect, regrets_gauss[0], label='gauss', color='magenta')
        plt.fill_between(t_vect, regrets_gauss[1], regrets_gauss[2], color='magenta', alpha=phaal)
        plt.plot(t_vect, regrets_gumbel[0], label='gumbel', color='green')
        plt.fill_between(t_vect, regrets_gumbel[1], regrets_gumbel[2], color='green', alpha=phaal)'''
        '''plt.plot(t_vect, regrets_exp_2[0], label='exp_2', color='purple')
        plt.fill_between(t_vect, regrets_exp_2[1], regrets_exp_2[2], color='purple', alpha=phaal)'''
        '''plt.plot(t_vect, regrets_exp_high[0], label='exp high', color='green')
        plt.fill_between(t_vect, regrets_exp_high[1], regrets_exp_high[2], color='green', alpha=phaal)'''
        '''plt.plot(t_vect, regrets_exp_2_high[0], label='exp_2 high', color='grey')
        plt.fill_between(t_vect, regrets_exp_2_high[1], regrets_exp_2_high[2], color='grey', alpha=phaal)'''
        '''plt.plot(t_vect, regrets_exp_adapt[0], label='exp adaptive', color='yellow')
        plt.fill_between(t_vect, regrets_exp_adapt[1], regrets_exp_adapt[2], color='yellow', alpha=phaal)'''
        '''plt.plot(t_vect, regrets_gauss_adapt[0], label='gauss adaptive', color='grey')
        plt.fill_between(t_vect, regrets_gauss_adapt[1], regrets_gauss_adapt[2], color='grey', alpha=phaal)
        plt.plot(t_vect, regrets_gumbel_adapt[0], label='gumbel adaptive', color='blue')
        plt.fill_between(t_vect, regrets_gumbel_adapt[1], regrets_gumbel_adapt[2], color='blue', alpha=phaal)'''
        '''plt.plot(t_vect, regrets_exp_2_adapt[0], label='exp_2 adaptive', color='firebrick')
        plt.fill_between(t_vect, regrets_exp_2_adapt[1], regrets_exp_2_adapt[2], color='firebrick', alpha=phaal)'''
        '''plt.plot(t_vect, regrets_exp_adapt_high[0], label='exp adaptive high', color='blue')
        plt.fill_between(t_vect, regrets_exp_adapt_high[1], regrets_exp_adapt_high[2], color='blue', alpha=phaal)'''
        '''plt.plot(t_vect, regrets_exp_2_adapt_high[0], label='exp_2 adaptive high', color='magenta')
        plt.fill_between(t_vect, regrets_exp_2_adapt_high[1], regrets_exp_2_adapt_high[2], color='magenta', alpha=phaal)'''
        '''plt.plot(t_vect, regrets_exp_adapt_trunc[0], label='exp adaptive truncated', color='cyan')
        plt.fill_between(t_vect, regrets_exp_adapt_trunc[1], regrets_exp_adapt_trunc[2], color='cyan', alpha=phaal)'''
        # plt.plot(np.arange(T-T0+1)+T0-1, regrets_ucb1, label='CombUCB1', color='black')
        '''# plt.legend(loc='upper left')
        # Shrink current axis by 10%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("time")
        plt.ylabel("regret")'''

        '''plt.figure()
        plt.title("Bernoulli(0.5) losses with one best path(0.5-Delta), Delta = {}, grid size = {}".format(Delta, N))
        # plt.title("Bernoulli losses, grid size = {}".format(N))
        # plt.title("Bouncing random walk losses, sd = {}, grid size = {}".format(sd, N))
        plt.plot(t_vect, L)
        plt.xlabel("time")
        plt.ylabel("loss")'''
        """
        # plt.figure()
        fig2 = plt.figure()
        ax2 = plt.subplot(111)
        plt.title("FPL+RW vs CombUCB1 - Not bouncing random walk losses, sd = {}, grid size = {}".format(sd, N))
        plt.plot(t_vect, regrets2_exp[0], label='exp', color='red')
        plt.fill_between(t_vect, regrets2_exp[1], regrets2_exp[2], color='red', alpha=phaal)
        '''plt.plot(t_vect, regrets2_gauss[0], label='gauss', color='magenta')
        plt.fill_between(t_vect, regrets2_gauss[1], regrets2_gauss[2], color='magenta', alpha=phaal)
        plt.plot(t_vect, regrets2_gumbel[0], label='gumbel', color='green')
        plt.fill_between(t_vect, regrets2_gumbel[1], regrets2_gumbel[2], color='green', alpha=phaal)'''
        plt.plot(t_vect, regrets2_exp_2[0], label='exp_2', color='purple')
        plt.fill_between(t_vect, regrets2_exp_2[1], regrets2_exp_2[2], color='purple', alpha=phaal)
        plt.plot(t_vect, regrets2_exp_high[0], label='exp high', color='green')
        plt.fill_between(t_vect, regrets2_exp_high[1], regrets2_exp_high[2], color='green', alpha=phaal)
        plt.plot(t_vect, regrets2_exp_2_high[0], label='exp_2 high', color='grey')
        plt.fill_between(t_vect, regrets2_exp_2_high[1], regrets2_exp_2_high[2], color='grey', alpha=phaal)
        plt.plot(t_vect, regrets2_exp_adapt[0], label='exp adaptive', color='yellow')
        plt.fill_between(t_vect, regrets2_exp_adapt[1], regrets2_exp_adapt[2], color='yellow', alpha=phaal)
        '''plt.plot(t_vect, regrets2_gauss_adapt[0], label='gauss adaptive', color='grey')
        plt.fill_between(t_vect, regrets2_gauss_adapt[1], regrets2_gauss_adapt[2], color='grey', alpha=phaal)
        plt.plot(t_vect, regrets2_gumbel_adapt[0], label='gumbel adaptive', color='blue')
        plt.fill_between(t_vect, regrets2_gumbel_adapt[1], regrets2_gumbel_adapt[2], color='blue', alpha=phaal)'''
        plt.plot(t_vect, regrets2_exp_2_adapt[0], label='exp_2 adaptive', color='firebrick')
        plt.fill_between(t_vect, regrets2_exp_2_adapt[1], regrets2_exp_2_adapt[2], color='firebrick', alpha=phaal)
        plt.plot(t_vect, regrets2_exp_adapt_high[0], label='exp adaptive high', color='blue')
        plt.fill_between(t_vect, regrets2_exp_adapt_high[1], regrets2_exp_adapt_high[2], color='blue', alpha=phaal)
        plt.plot(t_vect, regrets2_exp_2_adapt_high[0], label='exp_2 adaptive high', color='magenta')
        plt.fill_between(t_vect, regrets2_exp_2_adapt_high[1], regrets2_exp_2_adapt_high[2], color='magenta', alpha=phaal)
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
        plt.title("Not bouncing random walk losses, sd = {}, grid size = {}".format(sd, N))
        plt.plot(t_vect, L2)
        plt.xlabel("time")
        plt.ylabel("loss")

        # plt.figure()
        fig3 = plt.figure()
        ax3 = plt.subplot(111)
        plt.title("FPL+RW vs CombUCB1 - Normal losses, sd = {}, grid size = {}".format(sd, N))
        plt.plot(t_vect, regrets3_exp[0], label='exp', color='red')
        plt.fill_between(t_vect, regrets3_exp[1], regrets3_exp[2], color='red', alpha=phaal)
        '''plt.plot(t_vect, regrets3_gauss[0], label='gauss', color='magenta')
        plt.fill_between(t_vect, regrets3_gauss[1], regrets3_gauss[2], color='magenta', alpha=phaal)
        plt.plot(t_vect, regrets3_gumbel[0], label='gumbel', color='green')
        plt.fill_between(t_vect, regrets3_gumbel[1], regrets3_gumbel[2], color='green', alpha=phaal)'''
        plt.plot(t_vect, regrets3_exp_2[0], label='exp_2', color='purple')
        plt.fill_between(t_vect, regrets3_exp_2[1], regrets3_exp_2[2], color='purple', alpha=phaal)
        plt.plot(t_vect, regrets3_exp_high[0], label='exp high', color='green')
        plt.fill_between(t_vect, regrets3_exp_high[1], regrets3_exp_high[2], color='green', alpha=phaal)
        plt.plot(t_vect, regrets3_exp_2_high[0], label='exp_2 high', color='grey')
        plt.fill_between(t_vect, regrets3_exp_2_high[1], regrets3_exp_2_high[2], color='grey', alpha=phaal)
        plt.plot(t_vect, regrets3_exp_adapt[0], label='exp adaptive', color='yellow')
        plt.fill_between(t_vect, regrets3_exp_adapt[1], regrets3_exp_adapt[2], color='yellow', alpha=phaal)
        '''plt.plot(t_vect, regrets3_gauss_adapt[0], label='gauss adaptive', color='grey')
        plt.fill_between(t_vect, regrets3_gauss_adapt[1], regrets3_gauss_adapt[2], color='grey', alpha=phaal)
        plt.plot(t_vect, regrets3_gumbel_adapt[0], label='gumbel adaptive', color='blue')
        plt.fill_between(t_vect, regrets3_gumbel_adapt[1], regrets3_gumbel_adapt[2], color='blue', alpha=phaal)'''
        plt.plot(t_vect, regrets3_exp_2_adapt[0], label='exp_2 adaptive', color='firebrick')
        plt.fill_between(t_vect, regrets3_exp_2_adapt[1], regrets3_exp_2_adapt[2], color='firebrick', alpha=phaal)
        plt.plot(t_vect, regrets3_exp_adapt_high[0], label='exp adaptive high', color='blue')
        plt.fill_between(t_vect, regrets3_exp_adapt_high[1], regrets3_exp_adapt_high[2], color='blue', alpha=phaal)
        plt.plot(t_vect, regrets3_exp_2_adapt_high[0], label='exp_2 adaptive high', color='magenta')
        plt.fill_between(t_vect, regrets3_exp_2_adapt_high[1], regrets3_exp_2_adapt_high[2], color='magenta', alpha=phaal)
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
        plt.title("Normal losses, sd = {}, grid size = {}".format(sd, N))
        plt.plot(t_vect, L3)
        plt.xlabel("time")
        plt.ylabel("loss")
        """

        """
        '''fig = plt.figure()
        ax = plt.subplot(111)
        plt.title("FPL+RW vs CombUCB1 - Bouncing random walk losses, sd = {}, grid size = {}".format(sd, N))
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
    ax1.plot(np.arange(T-T0+1)+T0-1, regrets_ucb1, label='CombUCB1', color='black')
    # ax2.plot(np.arange(T-T0+1)+T0-1, regrets_ucb1, label='CombUCB1', color='black')

    # plt.legend(loc='upper left')
    # Shrink current axis by 10%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    '''# plt.legend(loc='upper left')
    # Shrink current axis by 10%
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))'''

    plt.show()