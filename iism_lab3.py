import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iism_lab2 import a5_generator
from scipy.stats import chi2

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
            yield E + D * (sum([next(psi) for _ in range(n)]) - n / 2) / (n / 12) ** 0.5
        else:
            yield E + D * (-2 * np.log(next(psi))) ** 0.5 * np.sin(2 * np.pi * next(psi))


def exponential_psi(l=2):
    psi = uniform_psi(get_a5_generator(), n=10)
    while True:
        yield -np.log(next(psi)) / l


def polynomial_psi(ps):
    psi = uniform_psi(get_a5_generator(), n=10)
    ps = np.cumsum(ps)
    while True:
        val = next(psi)
        for i, p in enumerate(ps):
            if val < p:
                return i


def M(res):
    return np.mean(res)


def D(res):
    return np.std(res)


def empirical_distribution_func(res):
    ys, xs = np.histogram(res, bins=np.unique(res), density=True)
    return xs, np.cumsum(ys)


def empirical_histogram(res, xs):
    _min, _max = xs[0], xs[-1]
    ys = np.array([len([t for t in res if xs[i] <= t < xs[i + 1]]) for i in range(len(xs) - 1)]) / len(res)
    return xs, ys


def calc_function(func, xs, params=None):
    ys = [func(x, **params) if params is not None else func(x) for x in xs]
    return xs, ys


def draw_empirical_distribution_func(res, func=None, params=None, ax=None, label=None, loc=None):
    xs, ys = empirical_distribution_func(res)
    ax.bar(xs[:-1], ys, label=label, color='green')
    xs, ys = calc_function(func, xs[:-1], params)
    ax.plot(xs, ys, color='blue')
    ax.legend(loc=loc)


def draw_empirical_histogram(res, xs, func=None, params=None, ax=None, label=None, loc=None):
    xs, ys = empirical_histogram(res, xs)
    ax.bar(xs[:-1], ys, label=label, color='green')
    xs, ys = calc_function(func, xs[:-1], params)
    ax.plot(xs, ys, color='blue')
    ax.legend(loc=loc)


def pearson_test(res, k, eps, func, *args):
    _min, _max, n = min(res), max(res), len(res)
    step = (_max - _min) / k
    x = np.array([val for val in np.arange(_min, _max, step)])
    y = np.array([len([t for t in res if val <= t < val + step]) for val in x]) / len(res)
    f = list(map(lambda t: func(t, *args), x)) + [func(_max + step, *args)]
    p = np.array(list(map(lambda x, y: x - y, f[:-1], f[1:])))
    xsi = np.sum((y - n * p) ** 2 / (n * p))
    P = 1 - chi2.cdf(xsi, k - 1)
    return eps < P


def kolmogorov_test(res, ):
    pass


if __name__ == '__main__':
    psi = exponential_psi()
    res = [next(psi) for _ in range(1000)]
    print(np.mean(res))
    print(np.std(res))
