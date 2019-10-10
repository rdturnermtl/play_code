from __future__ import print_function
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import chi, powerlaw

base_dist = chi


def mult0(a, b):
    if a == 0:
        return a
    return a * b


def log_logistic(x):
    v = np.minimum(x, 0.0) - np.log(1 + np.exp(-np.abs(x)))
    return v


def mix_logpdf(x, k, a, L):
    lp1 = base_dist.logpdf(x, df=k)
    lp2 = powerlaw.logpdf(x, a=k, loc=a, scale=L)

    logpdf = logsumexp([lp1, lp2]) + np.log(0.5)

    return logpdf


def KL_chi_mix(k, a, L):
    def integrand(x):
        v = base_dist.pdf(x, k) * (base_dist.logpdf(x, df=k) -
                                   mix_logpdf(x, k, a, L))
        return v
    KL, err = quad(integrand, 0.0, np.inf)
    return KL, err


def KL_unif_mix(k, a, L):
    def integrand(x):
        pp = powerlaw.pdf(x, a=k, loc=a, scale=L)
        v = pp * (powerlaw.logpdf(x, a=k, loc=a, scale=L) -
                  mix_logpdf(x, k, a, L))
        return v
    KL, err = quad(integrand, a, a + L)
    return KL, err


def JS_1(k, a, L):
    assert(k >= 0)
    assert(a >= 0)
    assert(L >= 0)

    KL1, err1 = KL_chi_mix(k, a, L)
    KL2, err2 = KL_unif_mix(k, a, L)
    JS = 0.5 * (KL1 + KL2)
    err = 0.5 * (err1 + err2)
    return JS, err


def JS_2(k, a, L):
    assert(k >= 0)
    assert(a >= 0)
    assert(L >= 0)

    def integrand(x):
        log_p = base_dist.logpdf(x, df=k)
        log_q = powerlaw.logpdf(x, a=k, loc=a, scale=L)
        delta = log_p - log_q
        v = mult0(np.exp(log_p), log_logistic(delta)) + \
            mult0(np.exp(log_q), log_logistic(-delta))
        return v
    JS_, err_ = quad(integrand, 0.0, np.inf)
    JS = np.log(2) + 0.5 * JS_
    err = 0.5 * err_
    return JS, err


def JS_3(k, a, L):
    assert(k >= 0)
    assert(a >= 0)
    assert(L >= 0)

    def integrand_1(x):
        log_p = base_dist.logpdf(x, df=k)
        log_q = powerlaw.logpdf(x, a=k, loc=a, scale=L)
        delta = log_p - log_q
        v = np.exp(log_p) * log_logistic(delta)
        return v

    def integrand_2(x):
        log_p = base_dist.logpdf(x, df=k)
        log_q = powerlaw.logpdf(x, a=k, loc=a, scale=L)
        delta = log_p - log_q
        v = np.exp(log_q) * log_logistic(-delta)
        return v

    P1, err1 = quad(integrand_1, 0.0, np.inf)
    P2, err2 = quad(integrand_2, a, a + L)
    JS = np.log(2) + 0.5 * (P1 + P2)
    err = 0.5 * (err1 + err2)
    return JS, err


JS = (JS_1, JS_2, JS_3)

if __name__ == '__main__':
    print(JS_1(2, 3, 2.5))
    print(JS_2(2, 3, 2.5))
    print(JS_3(2, 3, 2.5))

# TODO some more tests that all give same answer
# TODO test against MC estimate via error of bayes classifier
