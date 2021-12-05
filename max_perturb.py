# -*- coding: utf-8 -*-

from __future__ import division
from fpl_rw import *
from scipy.stats import gaussian_kde
from scipy.special import gammainc


num_dim = 100
m = 5
num = 100
# dim_vect = np.arange(num_dim) + 10
dim_vect = np.logspace(0, 3, num=num_dim) + m - 1
# y = np.zeros(num_dim)
# y_exp = np.zeros(num_dim)
max_exp = np.zeros(num_dim)
# y_gauss = np.zeros(num_dim)
max_gauss = np.zeros(num_dim)
# y_gumbel = np.zeros(num_dim)
max_gumbel = np.zeros(num_dim)
# y_exp_2 = np.zeros(num_dim)
sum_exp_2 = np.zeros(num_dim)
sum_exp_2_all = np.zeros(num_dim)
max_exp_2 = np.zeros(num_dim)
r_exp_2 = np.zeros(num_dim)
# var_exp_2 = np.zeros(num_dim)
# print np.var([perturb_vect(40, method="exp_2") for _ in range(10)])
for i, dim in enumerate(dim_vect):
    dim = int(np.ceil(dim))
    y = perturb_vect(dim, method="exp_2")
    # y = np.array(list(reversed(sorted(y))))
    # y_exp[i] = np.mean([max(perturb_vect(dim, method="exp")) for _ in range(num)])
    max_exp[i] = np.mean([max(perturb_vect(dim, method="exp")) for _ in range(num)])
    # y_gauss[i] = np.mean([max(perturb_vect(dim, method="gauss")) for _ in range(num)])
    max_gauss[i] = np.mean([max(perturb_vect(dim, method="gauss")) for _ in range(num)])
    # y_gumbel[i] = np.mean([max(perturb_vect(dim, method="gumbel")) for _ in range(num)])
    max_gumbel[i] = np.mean([max(perturb_vect(dim, method="gumbel")) for _ in range(num)])
    # y_exp_2[i] = np.mean([max(perturb_vect(int(np.floor(dim)), method="exp_2")) for _ in range(num)])
    # sum_exp_2[i] = np.mean([sum(np.abs(y[:m])) for _ in range(num)])
    # sum_exp_2_all[i] = np.mean([sum(np.abs(y)) for _ in range(num)])
    max_exp_2[i] = np.mean([max(y) for _ in range(num)])
    # r_exp_2[i] = np.mean([np.linalg.norm(y) for _ in range(num)])
    # var_exp_2[i] = np.var([perturb_vect(int(np.floor(dim)), method="exp_2") for _ in range(num)])
    print dim

fig = plt.figure()
ax = plt.subplot(111)
# plt.title("maxima of perturbations vectors")
plt.xlabel("dimension d")
# plt.plot(dim_vect, sum_exp_2, label='sum_exp_2')
# plt.plot(dim_vect, sum_exp_2_all, label='sum_exp_2_all')
plt.plot(dim_vect, max_exp_2, label='max_exp_2')
# plt.plot(dim_vect, r_exp_2, label='r_exp_2')
# plt.plot(dim_vect, var_exp_2, label='var_exp_2')
# plt.plot(dim_vect, y_exp, label='exp')
plt.plot(dim_vect, max_exp, label='max_exp')
# plt.plot(dim_vect, y_gauss, label='gauss')
plt.plot(dim_vect, max_gauss, label='max_gauss')
# plt.plot(dim_vect, y_gumbel, label='gumbel')
plt.plot(dim_vect, max_gumbel, label='max_gumbel')
# plt.plot(dim_vect, y_exp_2, label='exp_2')
# plt.plot(dim_vect, dim_vect, label='y=x')
# plt.plot(dim_vect, dim_vect**2, label='y=x**2')
plt.plot(dim_vect, np.sqrt(dim_vect), label='sqrt(x)')
plt.plot(dim_vect, np.log(dim_vect), label='log')
plt.plot(dim_vect, np.sqrt(np.log(dim_vect)), label='sqrt(log)')
plt.plot(dim_vect, np.log(np.log(dim_vect)), label='log(log)')
# plt.plot(dim_vect, np.sqrt(np.log(dim_vect)), label='sqrt(log)')
# plt.plot(dim_vect, np.log(np.log(dim_vect)), label='log(log)')
ax.set_xscale('log')
ax.set_yscale('log')
# plt.legend(loc='upper left')
# Shrink current axis by 10%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
'''
num_dim = 20
num = 100
dim_vect = np.arange(num_dim) + 1
d = dim_vect[-1]
R = 30
print "num = {}".format(num)
print "R = {}".format(R)
r_exp_2 = np.zeros(num_dim)
r_exp_2_m = np.zeros(num_dim)
cpt = np.zeros(num_dim)
cpt_m = np.zeros(num_dim)
for _ in range(num):
    for i, dim in enumerate(dim_vect):
        r_exp_2[i] = np.linalg.norm(perturb_vect(dim, method="exp_2"))
        r_exp_2_m[i] = np.linalg.norm(perturb_vect(d, method="exp_2")[:i+1])
        if r_exp_2[i] > R:
            cpt[i] += 1
        if r_exp_2_m[i] > R:
            cpt_m[i] += 1
def ball(dim, a, t):
    s = 0
    for i in range(dim):
        s += (a*t)**i/np.math.factorial(i)
    return 1 - s*np.exp(-a*t)
p1 = 1 - gammainc(d, R/2)/np.math.factorial(d-1)
p2 = 1 - ball(d, 1/2, R)
print 'p1 = {}'.format(p1)
print 'p2 = {}'.format(p2)
p_vect = np.zeros(num_dim)
for i in range(num_dim):
    # p_vect[i] = 1 - gammainc(dim_vect[i], R/2)/np.math.factorial(dim_vect[i]-1)
    p_vect[i] = 1 - ball(dim_vect[i], 1/2, R)
fig = plt.figure()
ax = plt.subplot(111)
plt.title("P( r > {} )".format(R))
plt.xlabel("dimension")
plt.plot(dim_vect, cpt/num, label='exp_2')
plt.plot(dim_vect, p_vect, label='exp_2 theory')
plt.plot(dim_vect, cpt_m/num, label='exp_2_m')
plt.plot(dim_vect, p2*dim_vect/d, label='P(R>{})*m/d'.format(R))
plt.plot(dim_vect, p2*np.ones(num_dim), label='P(R>{})'.format(R))
plt.plot(dim_vect, d/p2*cpt_m/num, label='unknown f(m, R)')
plt.plot(dim_vect, d*(1-np.exp(-dim_vect/R))/(1-np.exp(-d/R)), label='model for f(m, R)')
# plt.legend(loc='upper left')
# Shrink current axis by 10%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()'''
'''
dim = 10
num = 10000
R1 = np.random.rand() + 10
R2 = np.random.rand() + 20
R3 = np.random.rand() + 30
z_exp_2 = []
cpt1 = 0
cpt2 = 0
cpt3 = 0
for _ in range(num):
    v = perturb_vect_exp_2(dim)
    r = np.linalg.norm(v)
    if r < R1:
        cpt1 += 1
    if r < R2:
        cpt2 += 1
    if r < R3:
        cpt3 += 1
    # z_exp_2 = np.append(z_exp_2, perturb_vect_exp_2(dim))
# z_exp_2 = np.array([perturb_vect_exp_2(dim) for _ in range(100000)])
# print np.var(z_exp_2, ddof=1)
def ball(dim, a, t):
    s = 0
    for i in range(dim):
        s += (a*t)**i/np.math.factorial(i)
    return 1 - s*np.exp(-a*t)
print "dim = {}".format(dim)
print "num = {}".format(num)
print "R1 = {}".format(R1)
print "R2 = {}".format(R2)
print "R3 = {}".format(R3)
print "........for R1, presence frequency = {}, and ball(dim, 1/2, R1) = {}".format(cpt1/num, ball(dim, 1/2, R1))
print "........for R2, presence frequency = {}, and ball(dim, 1/2, R2) = {}".format(cpt2/num, ball(dim, 1/2, R2))
print "........for R3, presence frequency = {}, and ball(dim, 1/2, R3) = {}".format(cpt3/num, ball(dim, 1/2, R3))'''

'''density = gaussian_kde(z_exp_2)
x = np.linspace(0, 10, 1000)
values = density(x)'''

'''fig = plt.figure()
ax = plt.subplot(111)
plt.hist(z_exp_2, bins=1000, normed=True, alpha=0.1)
plt.plot(x, values, label='kernel estimation')
for u in np.linspace(2, 3, 5):
    plt.plot(x, 1/2*u*np.exp(-u*x), label='u={}'.format(u))
# ax.set_xscale('log')
ax.set_yscale('log')
# plt.legend(loc='upper left')
# Shrink current axis by 10%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()'''