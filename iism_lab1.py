import sys, argparse
import numpy as np
import matplotlib.pyplot as plt


def linear_generator(x0, a, c, M):
    xn = x0
    while True:
        yield xn
        xn = (a * xn + c) % M


def M_generator(gen1, gen2, M = 2 ** 10 + 1, k=64):
    v = [next(gen1) for _ in range(k)]
    while True:
        x, y = next(gen1), next(gen2)
        j = int(y * k / M)
        yield v[j]
        v[j] = x


def normally_psi(gen, M):
    while True:
        yield next(gen) / M


def discrete_psi(gen, j):
    while True:
        yield next(gen) % j


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--mod_normally', action='store_true', dest='mod')
    args = parser.parse_args()
    gen1 = linear_generator(2, 1, 100, 101)
    gen2 = linear_generator(5, 1, 2 ** 10, 2 ** 10 + 1)
    gen3 = M_generator(gen1, gen2, 2 ** 10 + 1, 64)
    gen4 = M_generator(gen1, gen2, 2 ** 10 + 1, 256)
    gens = [gen1, gen2, gen3, gen4]
    maxs = [101, 2 ** 10 + 1, 101, 101]

    if args.mod:
        all_examples = []
        for i in range(4):
            gen = normally_psi(gens[i], maxs[i])
            all_examples.append(np.array([next(gen) for _ in range(5000)]))

        fig = plt.figure()
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.hist(all_examples[i])

        fig = plt.figure()
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.scatter(all_examples[i][:-1], all_examples[i][1:])
    else:
        all_examples = []
        for i in range(4):
            for j in range(2, 5):
                gen = discrete_psi(gens[i], j)
                all_examples.append(np.array([next(gen) for _ in range(5000)]))

        fig = plt.figure()
        for i in range(4):
            for j in range(2, 5):
                plt.subplot(4, 3, i * 3 + j - 1)
                plt.hist(all_examples[i * 3 + j - 2], j)

    fig = plt.figure()
    for i, examples in enumerate(all_examples):
        begin_moments = [np.mean(examples),
                         np.mean(examples ** 2),
                         np.mean(examples ** 3),
                         np.mean(examples ** 4),
                         np.mean(examples ** 5)]
        E = np.mean(examples)
        centre_moments = [np.mean((examples - E)),
                          np.mean((examples - E) ** 2),
                          np.mean((examples - E) ** 3),
                          np.mean((examples - E) ** 4),
                          np.mean((examples - E) ** 5)]
        ax = fig.add_subplot(4, 3, i + 1)
        ax.text(10, 60, 'Moments:', fontsize=8)
        for i, val in enumerate(begin_moments):
            ax.text(10, 50 - i * 10, str(val), fontsize=8)
        ax.text(100, 60, 'Normalized moments', fontsize=8)
        for i, val in enumerate(centre_moments):
            ax.text(100, 50 - i * 10, str(val), fontsize=8)
        ax.axis([0, 200, 0, 75])
    plt.show()
