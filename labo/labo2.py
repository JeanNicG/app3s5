import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.fft as fft

fc = 2000
fe = 16000

N1 = 16
N2 = 32
N3 = 64

def k_calc(N):
    K = 2*N*fc/fe + 1
    print(K)
    return K

K1 = k_calc(N1)
K2 = k_calc(N2)
K3 = k_calc(N3)

def h_calc(N, K):
    h = np.zeros(N)
    h[0] = K/N
    for k in range(1, N):
        h[k] = (1/N) * (np.sin(np.pi*k*K/N) / np.sin(np.pi*k/N))
    return h

h1 = h_calc(N1, K1)
h2 = h_calc(N2, K2) 
h3 = h_calc(N3, K3)

h1_shift = fft.fftshift(h1) * np.hamming(len(h1))
h2_shift = fft.fftshift(h2) * np.hamming(len(h2))
h3_shift = fft.fftshift(h3) * np.hamming(len(h3))


plt.figure()
plt.stem(range(len(h1_shift)), h1_shift, label='N=16')
plt.figure()
plt.stem(range(len(h2_shift)), h2_shift, label='N=32')
plt.figure()
plt.stem(range(len(h3_shift)), h3_shift, label='N=64')
plt.title("Réponse impulsionnelle des filtres passe-bas")
plt.show()