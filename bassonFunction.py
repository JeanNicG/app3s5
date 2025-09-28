import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

def get_lowPass(fc, fe, order):
    wc = 2 * np.pi * fc/fe
    n = np.arange(order)
    n_shift = n - (order - 1) / 2
    h = np.sinc(wc * n_shift / np.pi) * (wc / np.pi)
    h = h * np.hamming(order)
    return h

def get_stopBand(w0, w1, fe, order):
    lp_filter1 = get_lowPass(w0-w1, fe, order)
    lp_filter2 = get_lowPass(w0+w1, fe, order)
    sb_filter = dirac_delta(order) - (lp_filter2 - lp_filter1)
    return sb_filter

def dirac_delta(N):
    delta = np.zeros(N)
    delta[N//2] = 1
    return delta

def apply_filter(audio_data, filtre, N):
    for i in range(N):
        audio_data = signal.convolve(audio_data, filtre, mode='same')
    return audio_data