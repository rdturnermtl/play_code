import numpy as np
from scipy.optimize import minimize_scalar
import high_gauss as hg

from matplotlib import rcParams, use
#use('pdf')
import matplotlib.pyplot as plt  # noqa: E402, mpl gives no other choice :(

# Matplotlib setup
# Note this will put type-3 font BS in the pdfs, if it matters
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'

ext = '.png'
D = (2, 3, 10, 30, 100, 1000)
colors = ('r', 'g', 'b', 'c', 'm', 'k')


def JS(k, a, b, mode=0):
    JS_, _ = hg.JS[mode](k, a, b - a)
    return JS_


def opt_a(k, b):
    f = lambda a: JS(k, a, b)  # noqa E731, dumb rule
    res = minimize_scalar(f, bracket=(0.0, b))
    a = res.x
    return a


def opt_b(k, a):
    assert(a < np.sqrt(k))

    bounds = (a, np.sqrt(k) + 1.0) if k <= 20 else \
        (np.sqrt(k), np.sqrt(k) + 0.33)

    f = lambda b: JS(k, a, b)  # noqa E731, dumb rule
    res = minimize_scalar(f, bounds=bounds, method='bounded')
    b = res.x
    return b


# find optimal b
opt_b_by_D = {dim: opt_b(dim, a=0) for dim in D}

for dim in D:
    # then show optimal a as x-section
    bounds = (0, np.sqrt(dim) + 1.0) if dim <= 20 else \
        (np.sqrt(dim), np.sqrt(dim) + 0.33)
    b_grid = np.linspace(bounds[0], bounds[-1], 250)
    JS_vals = [JS(dim, a=0, b=bb) for bb in b_grid]

    b_star = opt_b_by_D[dim]

    plt.figure()
    plt.plot(b_grid, JS_vals, '.-')
    plt.plot(b_star, JS(dim, a=0, b=b_star), 'ro')
    lower, upper = plt.ylim()
    plt.vlines(np.sqrt(dim), lower, upper)
    plt.xlabel('$b$')
    plt.ylabel('JS')
    plt.title('Gauss-uniform divergence by outer radius in $D = %d$' % dim)
    plt.grid('on')
    plt.savefig('opt_b_%d%s' % (dim, ext), dpi=300)

    a_grid = np.linspace(0, 0.9 * b_star, 250)
    JS_vals = [JS(dim, a=aa, b=b_star) for aa in a_grid]

    plt.figure()
    plt.plot(a_grid, JS_vals, '.-')
    plt.xlabel('$a$')
    plt.ylabel('JS')
    plt.title('Gauss-uniform divergence by inner radius in $D = %d$' % dim)
    plt.grid('on')
    plt.savefig('opt_a_%d%s' % (dim, ext), dpi=300)

# show optimal b as function of D
x = np.arange(1, 101)
bs = [opt_b(dim, a=0) for dim in x]
plt.figure()
plt.plot(x, bs, 'b.-', label='best radius')
plt.plot(x, np.sqrt(x), 'k--', label='sqrt')
plt.xlabel('$D$')
plt.ylabel('$b$')
plt.legend()
plt.title('JS optimal outer radius by dimension')
plt.grid('on')
plt.savefig('b_by_D' + ext, dpi=300)
