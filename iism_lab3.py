import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iism_lab2 import a5_generator


# def normal_psi(gen, depth):
#     while True:
#         n1, n2, n3 = 0, 1, 1
#         m1, m2, m3 = 1, 2, 1
#         for _ in range(depth):
#             if next(gen) == 1:
#                 n1, n2, n3 = n2, n2 + n3, n3
#                 m1, m2, m3 = m2, m2 + m3, m3
#             else:
#                 n1, n2, n3 = n1, n1 + n2, n2
#                 m1, m2, m3 = m1, m1 + m2, m2
#         yield n2 / m2


def uniform_psi(gen, n):
    while True:
        yield int(''.join([str(next(gen)) for _ in range(n)]), 2) / (2 ** n - 1)


def poisson_psi(l=2.718284590452, x=300):
    S19 = np.random.randint(2, size=19)
    S22 = np.random.randint(2, size=22)
    S23 = np.random.randint(2, size=23)
    gen = a5_generator(S19, S22, S23)
    psi = uniform_psi(gen, n=10)
    p = l / x
    while True:
        k = 0
        for _ in range(x):
            if next(psi) < p:
                k += 1
        yield k


def normal_psi(n, a=0, s=1):
    S19 = np.random.randint(2, size=19)
    S22 = np.random.randint(2, size=22)
    S23 = np.random.randint(2, size=23)
    gen = a5_generator(S19, S22, S23)
    psi = uniform_psi(gen, n=10)
    while True:
        yield a + s * (sum([next(psi) for _ in range(n)]) - n / 2) / (n / 12) ** 0.5


def exponential_psi(l):
    pass

if __name__ == '__main__':
    psi = normal_psi(10)
    res = [next(psi) for _ in range(1000)]
    print(res)
    print(np.mean(res))
    print(np.std(res))
