import numpy as np
from scipy.stats import chi, powerlaw
from scipy.stats import gaussian_kde
from scipy.stats import kurtosis, iqr
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


def bilog10(x):
    y = np.sign(x) * np.log10(1.0 + np.abs(x))
    return y


def kdeplot(x, lower=-np.inf, upper=np.inf):
    x_grid = np.linspace(max(lower, np.min(x)), min(upper, np.max(x)), 1000)
    k = gaussian_kde(x)
    dd = k(x_grid)
    return x_grid, dd


# plot distn of r for diff D, also do same for uniform ball
x = np.linspace(0, 2.0, 1000)

plt.figure()
for dim, color in zip(D, colors):
    pdf = chi.pdf(np.sqrt(dim) * x, df=dim)
    plt.plot(x, pdf / np.max(pdf), color, label='$D=%d$' % dim)
plt.xlabel('$r$ (rescaled)')
plt.ylabel('$\chi$ pdf (rescaled)')
plt.legend()
plt.title('distributions on radius in gaussians')
plt.savefig('fig0' + ext, dpi=300)

plt.figure()
for dim, color in zip(D, colors):
    pdf = powerlaw.pdf(np.sqrt(dim) * x, a=dim, loc=0, scale=np.sqrt(dim))
    plt.plot(x, pdf / np.max(pdf), color, label='$D=%d$' % dim)
plt.xlabel('$r$ (rescaled)')
plt.ylabel('power law pdf (rescaled)')
plt.legend()
plt.title('distributions on radius in uniform ball')
plt.savefig('fig1' + ext, dpi=300)

# warp 2D gauss to look like diff D
n = 500

plt.figure()
for dim, color in zip(D, colors):
    R = np.sqrt(np.random.chisquare(df=dim, size=n))
    R = R / np.median(R)

    X = np.random.randn(n, 2)
    # Normalize to R=1
    X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    X = X * R[:, None]

    plt.plot(X[:, 0], X[:, 1], color + '.', label='D=%d' % dim)
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.legend()
plt.title('2D Gaussian with radius warped to $D$')
plt.savefig('fig2' + ext, dpi=300)

# Warp D gauss to look like 2D gauss then plot 2D marginals
n = 5000

plt.figure()
for dim, color in zip(D, colors):
    if dim < 2:
        continue
    R = np.sqrt(np.random.chisquare(df=2, size=n))
    R = R / np.median(R)

    X = np.random.randn(n, dim)
    # Normalize to R=1
    X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    X = X * R[:, None]

    # Now get marginals
    kurt = kurtosis(X[:, 0])
    xx, dd = kdeplot(X[:, 0] / np.median(np.abs(X[:, 0])))
    plt.plot(xx, dd, color, label='D=%d, K=%f' % (dim, kurt))
plt.xlabel('$x_0$ rescaled')
plt.ylabel('pdf')
plt.legend()
plt.title('marginal of $D$ gaussian with radius warped to 2D')
plt.savefig('fig3' + ext, dpi=300)

# warp 2D gauss to look like diff D
n = 500

# warp 2D gauss to look like diff D in likelihood
fig, axes = plt.subplots(nrows=1, ncols=2)
for dim, color in zip(D, colors):
    logR = np.log(np.sqrt(np.random.chisquare(df=dim, size=n))) * (float(dim) / 2)
    R = np.exp(logR - np.median(logR))

    X = np.random.randn(n, 2)
    # Normalize to R=1
    X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    X = X * R[:, None]

    axes[0].plot(X[:, 0], X[:, 1], color + '.', label='D=%d' % dim, zorder=1.0/dim)
    axes[1].plot(bilog10(X[:, 0]), bilog10(X[:, 1]), color + '.', label='D=%d' % dim, zorder=1.0/dim)
axes[0].set_xlim(-30, 30)
axes[0].set_ylim(-30, 30)
axes[0].set_xlabel('$x_0$')
axes[0].set_ylabel('$x_1$')
axes[1].set_xlabel('bilog $x_0$')
axes[1].set_ylabel('bilog $x_1$')
plt.legend()
plt.title('2D Gaussian with likelihood warped to $D$')
plt.savefig('fig4' + ext, dpi=300)

# warp 2D gauss to look like diff D in likelihood
n = 50000

fig, axes = plt.subplots(nrows=1, ncols=2)
for dim, color in zip(D, colors):
    logR = np.log(np.sqrt(np.random.chisquare(df=dim, size=n))) * (float(dim) / 2)
    R = np.exp(logR - np.median(logR))

    X = np.random.randn(n, 2)
    # Normalize to R=1
    X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    X = X * R[:, None]

    # Now get marginals
    kurt = kurtosis(X[:, 0])
    scale = iqr(X[:, 0])
    xx, dd = kdeplot(X[:, 0] / scale, lower=-10, upper=10)
    axes[0].plot(xx, dd, color, label='D=%d, K=%f' % (dim, kurt))
    axes[1].plot(sorted(X[:, 0] / scale), np.linspace(0.0, 1.0, len(X)),
                 color, label='D=%d, K=%f' % (dim, kurt))
axes[0].set_xlim(-3, 3)
axes[1].set_xlim(-1, 1)
axes[1].set_ylim(0, 1)
axes[0].set_xlabel('$x_0$')
axes[0].set_ylabel('pdf')
axes[1].set_xlabel('$x_0$')
axes[1].set_ylabel('cdf')
axes[1].legend()
plt.title('marginal of 2D Gaussian\nwith likelihood warped to $D$')
plt.savefig('fig5' + ext, dpi=300)

# Warp D gauss to look like 2D gauss in likelihood

plt.figure()
for dim, color in zip(D, colors):
    logR = np.log(np.sqrt(np.random.chisquare(df=dim, size=n))) * (2 / float(dim))
    R = np.exp(logR - np.median(logR))
    #R = np.sqrt(np.random.chisquare(df=2, size=n)) ** (2 / float(dim))
    #R = R / np.median(R)

    X = np.random.randn(n, dim)
    # Normalize to R=1
    X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    X = X * R[:, None]

    # Now get marginals
    kurt = kurtosis(X[:, 0])
    xx, dd = kdeplot(X[:, 0] / np.median(np.abs(X[:, 0])))
    plt.plot(xx, dd, color, label='D=%d, K=%f' % (dim, kurt))
plt.xlabel('$x_0$ rescaled')
plt.ylabel('pdf')
plt.legend()
plt.title('marginal of $D$ gaussian likelihood warped to 2D')
plt.savefig('fig6' + ext, dpi=300)
