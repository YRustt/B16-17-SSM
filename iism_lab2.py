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

def universal(seq, **kwargs):
    def to_number(arr):
        return np.packbits(arr, axis=-1) >> 2
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
    p_value = erfc(abs((f - kwargs['expectedValue']) / (pow(2, 0.2) * kwargs['sigma'])))
    return p_value

def test_universal(seq):
    n = 400000
    L = 6
    Q = 640
    K = int(n / L) - Q
    expectedValue = 5.2177052
    variance = 2.954
    c = 0.7 - 0.8 / L + (4 + 32 / L) / 15 * pow(K, -3 / L)
    sigma = c * pow(variance / K, 0.5)
    p_value = universal(seq, n=n, L=L, Q=Q, K=K,
                        expectedValue=expectedValue, sigma=sigma)
    return p_value

def linear_complexity(seq, M, n):
    def berlekamp_massey_alg(s):
        n = len(s)
        b, c = np.zeros(n), np.zeros(n)
        b[0], c[0] = 1, 1
        L, m = 0, -1
        for N in range(n):
            sum = 0
            for i in range(1, L + 1):
                sum += c[i] * s[N - i]
            d = np.mod(sum + s[N], 2)
            if d == 0:
                continue
            t = np.copy(c)
            for i in range(N - m, n):
                c[i] = np.mod(c[i] + b[i - N + m], 2)
            if 2 * L <= N:
                L, m = N + 1 - L, N
                b = np.copy(t)
        for i in range(len(c) - 1, -1, -1):
            if c[i] == 1:
                c = c[:i + 1]
                break
        return c
    N = n // M
    L = [len(berlekamp_massey_alg(seq[i * M: (i + 1) * M])) for i in range(N)]
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
            phi_m += c * np.log2(c)
    phi_m_plus = 0
    for c in C_m_plus:
        if c != 0:
            phi_m_plus += c * np.log2(c)
    ap_en = phi_m - phi_m_plus
    xsi = 2 * n * int(np.log(2) - ap_en)
    p_value = gammaincc(2 ** (m - 1), xsi / 2)
    return p_value

if __name__=='__main__':
    S = np.random.randint(2, size=32)
    c = np.array([1 if x in [0, 10, 30, 31] else 0 for x in range(32)])
    lfsr = lfsr_generator(S, c)
    ss = ss_generator(lfsr)
    seq_lfsr = [next(lfsr) for _ in range(400000)]
    seq_ss = [next(ss) for _ in range(400000)]

    gen1 = linear_generator(2, 1, 100, 101)
    gen2 = linear_generator(5, 1, 2 ** 32, 2 ** 32 + 1)
    gen3 = M_generator(gen1, gen2, 2 ** 32 + 1, 64)
    gen4 = M_generator(gen1, gen2, 2 ** 32 + 1, 256)
    seq_dpsi1 = [discrete_psi(gen1, 2) for _ in range(400000)]
    seq_dpsi2 = [discrete_psi(gen2, 2) for _ in range(400000)]
    seq_dpsi3 = [discrete_psi(gen3, 2) for _ in range(400000)]
    seq_dpsi4 = [discrete_psi(gen4, 2) for _ in range(400000)]

    print('Universal:')
    print(test_universal(seq_lfsr))
    print(test_universal(seq_ss))
    print(test_universal(seq_dpsi1))
    print(test_universal(seq_dpsi2))
    print(test_universal(seq_dpsi3))
    print(test_universal(seq_dpsi4))

    print('Linear complexity:')
    print(linear_complexity(seq_lfsr, M=1000, n=400000))
    print(linear_complexity(seq_ss, M=1000, n=400000))
    print(linear_complexity(seq_dpsi1, M=1000, n=400000))
    print(linear_complexity(seq_dpsi2, M=1000, n=400000))
    print(linear_complexity(seq_dpsi3, M=1000, n=400000))
    print(linear_complexity(seq_dpsi4, M=1000, n=400000))

    print('Approximate entropy:')
    print(approximate_entropy(seq_lfsr, m=6, n=400000))
    print(approximate_entropy(seq_ss, m=6, n=400000))
    print(approximate_entropy(seq_dpsi1, m=6, n=400000))
    print(approximate_entropy(seq_dpsi2, m=6, n=400000))
    print(approximate_entropy(seq_dpsi3, m=6, n=400000))
    print(approximate_entropy(seq_dpsi4, m=6, n=400000))

