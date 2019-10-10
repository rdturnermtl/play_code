import numpy as np
from scipy.stats import norm, chi
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde

from matplotlib import rcParams, use
#use('pdf')
import matplotlib.pyplot as plt  # noqa: E402, mpl gives no other choice :(

# Matplotlib setup
# Note this will put type-3 font BS in the pdfs, if it matters
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'


def cdfplot(x):
    plt.plot(sorted(x), np.linspace(0.0, 1.0, len(x)), '.-')


def kdeplot(x):
    x_grid = np.linspace(np.min(x), np.max(x), 1000)
    k = gaussian_kde(x)
    dd = k(x_grid)
    return x_grid, dd


ext = '.png'

n = 100000
k1 = 2
k2 = 5

np.random.seed(345655)

R = np.sqrt(np.random.chisquare(df=k2, size=n)) ** (float(k2) / k1)
# With jacobian adjustment for **k2/k1, ignoring additive constant
logp_R = chi.logpdf(R ** (float(k1) / k2), df=k2) + \
    ((float(k1) / k2) - 1) * np.log(R)

X = np.random.randn(n, k1)
# Normalize to R=1
X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
X = X * R[:, None]

# Adjustment for transform to cartesian from just radius, propto 1/powerlaw to
# transform powerlaw to uniform, ignoring additive constant
logp_X = logp_R - (k1 - 1) * np.log(R)
nll1 = -logp_X

X2 = np.random.randn(n, k2)
nll2 = -np.sum(norm.logpdf(X2), axis=1)

_, pval = ks_2samp(nll1 - np.median(nll1), nll2 - np.median(nll2))
print pval

plt.figure()
xx, dd = kdeplot(nll1 - np.min(nll1))
plt.plot(xx, dd, 'r', label='likelihood warped')
xx, dd = kdeplot(nll2 - np.min(nll2))
plt.plot(xx, dd, 'g', label='%dD gaussian' % k2)

# Standard gauss likelihood
X2 = np.random.randn(n, k1)
nll2 = -np.sum(norm.logpdf(X2), axis=1)
xx, dd = kdeplot(nll2 - np.min(nll2))
plt.plot(xx, dd, 'b', label='%dD gaussian' % k1)

# Now radious warped only
R = np.sqrt(np.random.chisquare(df=k2, size=n))
logp_R = chi.logpdf(R, df=k2)

X = np.random.randn(n, k1)
# Normalize to R=1
X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
X = X * R[:, None]

# Adjustment for transform to cartesian from just radius, propto 1/powerlaw to
# transform powerlaw to uniform, ignoring additive constant
logp_X = logp_R - (k1 - 1) * np.log(R)
nll1 = -logp_X

xx, dd = kdeplot(nll1 - np.min(nll1))
plt.plot(xx, dd, 'k', label='radius warped')

plt.xlabel('nll (shifted)')
plt.ylabel('pdf')
plt.legend()
plt.title('likelihood on likelihood at $D=%d$' % k2)
plt.savefig('fig3b' + ext, dpi=300)
