import numpy as np
from scipy.special import gamma
from scipy.stats import chi
from scipy.stats import ksone


def sphere_vol(r, k):
    V = ((np.pi ** (0.5 * k)) / gamma(0.5 * k + 1)) * (r ** k)
    return V


def shell_vol(r1, r2, k):
    V = sphere_vol(r2, k) - sphere_vol(r1, k)
    return V


def get_r_max(k, eps_mass):
    r_max = chi.isf(eps_mass, df=k)
    return r_max


def get_r_grid(k, n, r_max):
    r_grid = np.linspace(0, r_max ** k, n) ** (1.0 / k)
    return r_grid


def get_bin_prob(k, r_grid):
    cdf = chi.cdf(r_grid, df=k)
    bin_prob = np.diff(cdf)
    return bin_prob


n = 10000
eps_mass = 0.001

k1 = 2
k2 = 100

r_grid1 = get_r_grid(k1, n, get_r_max(k1, eps_mass))
r_grid2 = get_r_grid(k2, n, get_r_max(k2, eps_mass))

bin_prob = get_bin_prob(k2, r_grid2)
chi_pdf = chi.pdf(r_grid2[:-1], df=k2)

r_grid2B = r_grid1 ** (float(k1)/k2)
c = r_grid2[-1] / r_grid2B[-1]

import matplotlib.pyplot as plt  # noqa: E402, mpl gives no other choice :(
plt.plot(r_grid2[:-1], bin_prob / np.diff(r_grid2), 'b.-')
plt.plot(r_grid2[:-1], chi_pdf, 'r.-')

plt.figure()
plt.plot(r_grid1[:-1], bin_prob / np.diff(r_grid1), 'b.-')
chi_pdf = chi.pdf(c * r_grid1[:-1] ** (float(k1)/k2), df=k2)
plt.plot(r_grid1[:-1], chi_pdf * (np.diff(r_grid2) / np.diff(r_grid1)), 'r.-')


#x = np.arange(0,len(r_grid2))
#c = get_r_max(k2, eps_mass)**k2 / (n-1)
#err = np.max(np.abs(r_grid2 - (c*x)**(1.0/k2)))
#err = np.max(np.abs(np.diff(r_grid2) - ((c*x[1:])**(1.0/k2) - (c*x[:-1])**(1.0/k2))))

plt.figure()
X = np.random.randn(100000,k2)
R = np.sqrt(np.sum(X**2,axis=1))
R = R[R < get_r_max(k2, eps_mass)]
cc=(get_r_max(k2, eps_mass)**k2/get_r_max(k1, eps_mass)**k1)**(1.0/k2)
RR = (R/cc) ** (float(k2)/k1)

plt.plot(r_grid1[:-1],np.cumsum(bin_prob)/np.sum(bin_prob),'b.-')
ep=ksone.isf(.05,n=len(RR))
plt.plot(sorted(RR),np.linspace(0,1,len(RR)),'r.-')
plt.plot(sorted(RR),np.linspace(0,1,len(RR))-ep,'r.-')
plt.plot(sorted(RR),np.linspace(0,1,len(RR))+ep,'r.-')
