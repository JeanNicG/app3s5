import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import bassonFunction as bf
import guitarFunction as sf

fe, audio_data = wavfile.read('note_basson_plus_sinus_1000_hz.wav')
N = 6000
w0 = 1000
w1 = 40

sb_filter = bf.get_stopBand(w0, w1, fe, N)

enveloppe = sf.get_enveloppe(audio_data, sb_filter)
filtered_audio = bf.apply_filter(audio_data, sb_filter, 10)

filtered_audio = filtered_audio.astype(audio_data.dtype)
wavfile.write('filtered_audio.wav', fe, filtered_audio)

# Time domain plot
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.arange(N), sb_filter)
plt.title("Filtre coupe-bande (domaine temporel)")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid()

# Frequency domain plot
freq = np.fft.fftfreq(N, d=1/fe)
freq_response = np.fft.fft(sb_filter)
plt.subplot(2, 1, 2)
plt.plot(freq[:N//2], 20*np.log10(np.abs(freq_response[:N//2])))
plt.title("Réponse en fréquence du filtre coupe-bande")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.tight_layout()
plt.show()