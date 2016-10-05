import matplotlib.pyplot as plt
import numpy as np
from math import pow
from scipy.special import gammaincc, erfc

from iism_lab1 import linear_generator, M_generator, discrete_psi

def lfsr_generator(S, c):
    while True:
        yield S[0]
        S = np.append(S, np.mod(np.sum(S * c), 2))[1:]


def ss_generator(lfsr):
    a, b = next(lfsr), next(lfsr)
    while True:
        if a == 1:
            yield b
        a, b = b, next(lfsr)

def a5_generator(S19, S22, S23):
    c19 = np.array([1 if x in [0, 1, 2, 5] else 0 for x in range(19)])
    c22 = np.array([1 if x in [0, 1] else 0 for x in range(22)])
    c23 = np.array([1 if x in [0, 1, 2, 15] else 0 for x in range(23)])
    while True:
        F = 1 if S19[10] + S22[11] + S23[12] > 1 else 0
        yield (S19[0] + S22[0] + S23[0]) % 2
        if S19[10] == F:
            S19 = np.append(S19, np.mod(np.sum(S19 * c19), 2))[1:]
        if S22[11] == F:
            S22 = np.append(S22, np.mod(np.sum(S22 * c22), 2))[1:]
        if S23[12] == F:
            S23 = np.append(S23, np.mod(np.sum(S23 * c23), 2))[1:]


def universal(seq, **kwargs):
    def to_number(arr):
        return np.packbits(arr, axis=-1) >> 1
    L, Q, n, K = kwargs['L'], kwargs['Q'], kwargs['n'], kwargs['K']
    T = np.zeros(2 ** L, dtype=np.int32)
    for i in range(Q):
        try:
            j = to_number(seq[i * L: (i + 1) * L])
            T[j] = i
        except Exception:
            print(seq[i * L: (i + 1) * L])
    sum = 0
    for i in range(Q, Q + K):
        j = to_number(seq[i * L: (i + 1) * L])
        sum += np.log2(i - T[j])
        T[j] = i
    f = sum / K
    p_value = erfc(abs((f - kwargs['expectedValue']) / (pow(2, 0.5) * kwargs['sigma'])))
    return p_value


def test_universal(seq):
    n = 1000000
    L = 7
    Q = 1280
    K = int(n / L) - Q
    expectedValue = 6.1962507
    variance = 3.125
    c = 0.7 - 0.8 / L + (4 + 32 / L) / 15 * pow(K, -3 / L)
    sigma = c * pow(variance / K, 0.5)
    p_value = universal(seq, n=n, L=L, Q=Q, K=K,
                        expectedValue=expectedValue, sigma=sigma)
    return p_value


def linear_complexity(seq, M, n):
    N = n // M
    L = [berlekamp_massey_alg(seq[i * M: (i + 1) * M]) for i in range(N)]
    mu = M / 2 + (9 + pow(-1, M + 1)) / 36 - (M / 3 + 2 / 9) / pow(2, M)
    T = [pow(-1, M) * (l - mu) + 2 / 9 for l in L]
    v = np.zeros(7)
    for t in T:
        if t <= -2.5:
            v[0] += 1
        elif -2.5 < t <= -1.5:
            v[1] += 1
        elif -1.5 < t <= -0.5:
            v[2] += 1
        elif -0.5 < t <= 0.5:
            v[3] += 1
        elif 0.5 < t <= 1.5:
            v[4] += 1
        elif 1.5 < t <= 2.5:
            v[5] += 1
        else:
            v[6] += 1
    pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
    xsi = np.sum((v - N * pi) * (v - N * pi) / (N * pi))
    p_value = gammaincc(3, xsi / 2)
    return p_value


def berlekamp_massey_alg(s):
    n = len(s)
    b, c = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)
    b[0], c[0] = 1, 1
    L, m = 0, -1
    for N in range(n):
        d = (sum(c[j] * s[N - j] for j in range(1, L + 1)) + s[N]) % 2
        if d == 0:
            continue
        t = np.copy(c)
        for i in range(N - m, n):
            c[i] = (c[i] + b[i - N + m]) % 2
        if 2 * L <= N:
            L, m = N + 1 - L, N
            b = np.copy(t)
    return L


def approximate_entropy(seq, m, n):
    def to_number_from_m_bits(arr):
            return np.packbits(arr, axis=-1) >> (8 - m)
    def to_number_from_m_plus_bits(arr):
        return np.packbits(arr, axis=-1) >> (7 - m)
    seq_m = seq + seq[:m-1]
    seq_m_plus = seq + seq[:m]
    C_m = np.zeros(to_number_from_m_bits(np.ones(m, dtype=np.int8)) + 1)
    C_m_plus = np.zeros(to_number_from_m_plus_bits(np.ones(m + 1, dtype=np.int8)) + 1)
    for i in range(n):
        C_m[to_number_from_m_bits(seq_m[i: i+m])] += 1
        C_m_plus[to_number_from_m_plus_bits(seq_m_plus[i: i+m+1])] += 1
    C_m, C_m_plus = C_m / n, C_m_plus / n
    phi_m = 0
    for c in C_m:
        if c != 0:
            phi_m += c * np.log(c)
    phi_m_plus = 0
    for c in C_m_plus:
        if c != 0:
            phi_m_plus += c * np.log(c)
    ap_en = phi_m - phi_m_plus
    xsi = 2 * n * (np.log(2) - ap_en)
    p_value = gammaincc(2 ** (m - 1), xsi / 2)
    return p_value


def simple_test_approximate_entropy():
    seq = '1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000'
    seq = [int(s) for s in seq]
    m, n = 2, 100
    print(approximate_entropy(seq, m, n))


def simple_test_linear_complexity():
    with open('data/e.txt') as f:
        seq = [int(s) for s in f.read()]
        n, M = 1000000, 1000
        linear_complexity(seq, M, n)


def simple_test_berlekamp_massey_alg():
    seq = '11011101'
    seq = [int(s) for s in seq]
    c = berlekamp_massey_alg(seq)
    print('{}: {}'.format(seq, c))
    seq = '11001'
    seq = [int(s) for s in seq]
    c = berlekamp_massey_alg(seq)
    print('{}: {}'.format(seq, c))

def task1():
    S = np.random.randint(2, size=33)
    c = np.array([1 if x in [0, 10, 30, 31] else 0 for x in range(33)])
    lfsr = lfsr_generator(S, c)
    ss = ss_generator(lfsr)
    seq_lfsr = [next(lfsr) for _ in range(1000000)]
    seq_ss = [next(ss) for _ in range(1000000)]

    gen1 = discrete_psi(linear_generator(2, 1, 100, 101), 2)
    gen2 = discrete_psi(linear_generator(5, 1, 2 ** 32, 2 ** 32 + 1), 2)
    gen3 = discrete_psi(M_generator(gen1, gen2, 2 ** 32 + 1, 64), 2)
    gen4 = discrete_psi(M_generator(gen1, gen2, 2 ** 32 + 1, 256), 2)
    seq_dpsi1 = [next(gen1) for _ in range(1000000)]
    seq_dpsi2 = [next(gen2) for _ in range(1000000)]
    seq_dpsi3 = [next(gen3) for _ in range(1000000)]
    seq_dpsi4 = [next(gen4) for _ in range(1000000)]

    with open('data/e.txt') as f:
        seq = [int(s) for s in f.read()]

    with open('data/task_1_result1.txt', 'w') as f:
        print(1)
        f.write('Universal:\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(
                test_universal(seq_lfsr),
                test_universal(seq_ss),
                test_universal(seq_dpsi1),
                test_universal(seq_dpsi2),
                test_universal(seq_dpsi3),
                test_universal(seq_dpsi4),
                test_universal(seq)))
        print(2)
        f.write('Linear complexity:\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(
                linear_complexity(seq_lfsr, M=1000, n=1000000),
                linear_complexity(seq_ss, M=1000, n=1000000),
                linear_complexity(seq_dpsi1, M=1000, n=1000000),
                linear_complexity(seq_dpsi2, M=1000, n=1000000),
                linear_complexity(seq_dpsi3, M=1000, n=1000000),
                linear_complexity(seq_dpsi4, M=1000, n=1000000),
                linear_complexity(seq, M=1000, n=1000000)))
        print(3)
        f.write('Approximate entropy:\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(
                approximate_entropy(seq_lfsr, m=6, n=400000),
                approximate_entropy(seq_ss, m=6, n=400000),
                approximate_entropy(seq_dpsi1, m=6, n=400000),
                approximate_entropy(seq_dpsi2, m=6, n=400000),
                approximate_entropy(seq_dpsi3, m=6, n=400000),
                approximate_entropy(seq_dpsi4, m=6, n=400000),
                approximate_entropy(seq, m=6, n=400000)))
        print(4)

def task2():
    with open('data/task_2_result.txt', 'w') as f:
        S19 = np.random.randint(2, size=19)
        S22 = np.random.randint(2, size=22)
        S23 = np.random.randint(2, size=23)
        gen = a5_generator(S19, S22, S23)
        seq = [next(gen) for _ in range(1000000)]
        f.write('test for a5 generator:\n{}\n{}\n{}\n'.format(
                test_universal(seq),
                linear_complexity(seq, M=1000, n=1000000),
                approximate_entropy(seq, m=6, n=400000)))

if __name__ == '__main__':
    # task2()
    # task1()
    pass
