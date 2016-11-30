import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iism_lab2 import a5_generator
from scipy.stats import chi2
from math import gamma


def get_a5_generator():
    S19 = np.random.randint(2, size=19)
    S22 = np.random.randint(2, size=22)
    S23 = np.random.randint(2, size=23)
    gen = a5_generator(S19, S22, S23)
    return gen


def uniform_psi(gen, n=10):
    while True:
        yield (int(''.join([str(next(gen)) for _ in range(n)]), 2) + 1) / (2 ** n + 1)


def poisson_psi(l=1, x=200):
    psi = uniform_psi(get_a5_generator(), n=10)
    p = l / x
    while True:
        k = 0
        for _ in range(x):
            if next(psi) < p:
                k += 1
        yield k


def normal_psi(n, E=0, D=1):
    psi = uniform_psi(get_a5_generator(), n=10)
    while True:
        if n > 0:
            yield E + D * (sum([next(psi) for _ in range(n)]) - n / 2) * (12 / n) ** 0.5
        else:
            yield E + D * (-2 * np.log(next(psi))) ** 0.5 * np.sin(2 * np.pi * next(psi))


def exponential_psi(l=2):
    psi = uniform_psi(get_a5_generator(), n=10)
    while True:
        yield -np.log(next(psi)) / l


def polynomial_psi(n, ps):
    psi = uniform_psi(get_a5_generator(), n=10)
    ps = np.cumsum(ps)
    while True:
        res = []
        for i in range(n):
            val = next(psi)
            for i, p in enumerate(ps):
                if val < p:
                    res.append(i)
                    break
        yield np.histogram(res, bins=len(ps))[0]


def M(res):
    return np.mean(res)


def D(res):
    return np.std(res)


def empirical_histogram(res, left_border, right_border, k):
    step = (right_border - left_border) / k
    ys = np.array([len([t for t in res if xi <= t < xi + step])
                   for xi in np.arange(left_border, right_border, step)]) / len(res)
    return ys


def empirical_distribution_func(res, left_border, right_border, k):
    return np.cumsum(empirical_histogram(res, left_border, right_border, k))


def calc_function(func, xi, params=None):
    return func(xi, **params) if params is not None else func(xi)


def draw_empirical_distribution_func(res, func=None, params=None, type='bar', ax=None, label=None, loc=None):
    xs, ys = list(np.arange(-100, 101, 10)), empirical_distribution_func(res, -100, 100, 10)
    if type == 'bar':
        ax.bar(xs[:-1], ys, label=label, color='green')
    elif type == 'plot':
        ax.plot(xs[:-1], ys, label=label, color='green')
    ys = [calc_function(func, xi, params) for xi in xs]
    ax.plot(xs, ys, color='blue')
    ax.legend(loc=loc)


def draw_empirical_histogram(res, xs, func=None, params=None, ax=None, type='bar', label=None, loc=None):
    xs, ys = list(np.arange(-100, 101, 10)), empirical_histogram(res, -100, 100, 10)
    if type == 'bar':
        ax.bar(xs[:-1], ys, label=label, color='green')
    elif type == 'plot':
        ax.plot(xs[:-1], ys, label=label, color='green')
    ys = [calc_function(func, xi, params) for xi in xs]
    ax.plot(xs, ys, color='blue')
    ax.legend(loc=loc)


def pearson_test(res, xs, eps, func, params):
    def G(s, r):
        return 1 / (2 ** (r / 2) * gamma(r / 2)) * s ** (r / 2 - 1) * np.exp(-s / 2)
    n = len(res)
    ys = np.array(empirical_histogram(res, xs)[1])
    fs = np.array(calc_function(func, xs, params)[1])
    ps = np.array([fs[i + 1] - fs[i] for i in range(len(fs) - 1)])
    xsi = np.sum((ys - n * ps) * (ys - n * ps) / (n * ps))
    P = 1 - G(xsi, len(ps) - 1)
    return eps < P


def kolmogorov_test(res, eps, func, params):
    def K(t):
        return np.sum([(-1) ** j * np.exp(-2 * j ** 2 * t ** 2) for j in range(-100, 101)])
    n = len(res)
    xs, ys = empirical_distribution_func(res)
    dn = np.amax(np.array(calc_function(func, xs[:-1], params)[1]) - np.array(ys))
    P = 1 - K(n ** 0.5 * dn)
    return eps < P


if __name__ == '__main__':
    psi = exponential_psi()
    res = [next(psi) for _ in range(1000)]
    print(np.mean(res))
    print(np.std(res))

