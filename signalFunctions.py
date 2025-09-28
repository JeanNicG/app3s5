import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.fft as fft
from scipy.io import wavfile

def db_to_linear(db):
    return 10**(db/20)

def linear_to_db(linear):
    return 20 * np.log10(linear)

def N_calc(omega_c, target_gain):
    N = 0
    gain = 1
    print("target_gain:", target_gain)
    while gain > target_gain:
        N += 1
        gain = (1/N) * (np.sin(N*omega_c/2) /np.sin(omega_c/2)) # 3-51 lyons
    return N

def h_calc(N, omega_c):
    K = N*omega_c/np.pi + 1
    h = np.zeros(N)
    h[0] = 1/N
    k = 0
    for k in range(1, N):
        h[k] = (1/N) * (np.sin(np.pi*k*K/N) / np.sin(np.pi*k/N))
    return h

def get_enveloppe(audio_data, N_filtre):
    filtre = np.ones(N_filtre)/N_filtre
    enveloppe = np.convolve(audio_data, filtre)
    enveloppe = enveloppe / np.max(enveloppe)
    return enveloppe

def analyse_freq(audio_data, fe):
    audio_data_temporel = np.fft.fft(audio_data)
    amplitude_audio_data = np.abs(audio_data_temporel)
    freq_audio_data = np.fft.fftfreq(len(audio_data), 1/fe)
    
    # Only look at positive frequencies and skip DC component
    positive_frequencies = freq_audio_data[1:len(freq_audio_data)//2]
    positive_amplitudes = amplitude_audio_data[1:len(audio_data)//2]
    
    # Find the fundamental frequency
    note_la_d = np.argmax(positive_amplitudes)
    fondamental = positive_frequencies[note_la_d]
    
    harmonic = [np.abs(audio_data_temporel[i]) for i in range(0, 32)]
    phases = [np.angle(audio_data_temporel[i]) for i in range(0, 32)]
    
    return abs(fondamental), harmonic, phases

def note_dict(la_d):
    note_freq_dict = { 
        "DO":2**(-10/12) * la_d,
        "DO#":2**(-9/12) * la_d,
        "RE":2**(-8/12) * la_d,
        "RE#":2**(-7/12) * la_d,
        "MI":2**(-6/12) * la_d,
        "FA":2**(-5/12) * la_d,
        "FA#":2**(-4/12) * la_d,
        "SOL":2**(-3/12) * la_d,
        "SO#":2**(-2/12) * la_d,
        "LA": 2**(-1/12) * la_d,
        "LA#":la_d,
        "SI": 2**(1/12) * la_d
    }
    return note_freq_dict

def get_sound(harmonic, phases, fe, fondamental, enveloppe, duration):
    print("get_sound:", fondamental)
    t = np.linspace(0, duration, int(fe*duration))
    signal_synth = []
    
    # Use vectorized computation instead of loops
    for dt in t:
        val = 0
        for n in range(1, len(harmonic)):
            val += harmonic[n] * np.sin(2 * np.pi * n * fondamental * t - phases[n])
        signal_synth.append(val)

    # Resample envelope to match signal length
    x_original = np.linspace(0, 1, len(enveloppe))
    x_new = np.linspace(0, 1, len(signal_synth))
    enveloppe_resampled = np.interp(x_new, x_original, enveloppe)
    
    # Apply envelope by multiplication
    signal_synth = signal_synth * enveloppe_resampled
    
    return signal_synth

def get_silence(fe, duration):
    t = np.linspace(0, duration, int(fe*duration))
    silence = np.zeros_like(t)
    return silence

def composition_bethoven(harmonic, phases, fe, enveloppe, note_dict):
    sol = get_sound(harmonic, phases, fe, note_dict["SOL"], enveloppe, 0.25)
    mi = get_sound(harmonic, phases, fe, note_dict["MI"], enveloppe, 1)
    fa = get_sound(harmonic, phases, fe, note_dict["FA"], enveloppe, 0.25)
    re = get_sound(harmonic, phases, fe, note_dict["RE"], enveloppe, 1)
    
    silence = get_silence(fe, 0.1)
    silence1 = get_silence(fe, 0.5)

    musique = np.concatenate((
        sol, silence, 
        sol, silence, 
        sol, silence, 
        mi, silence1, 
        fa, silence, 
        fa, silence, 
        fa, silence, 
        re, silence1
    ))
    
    musique = np.int16(np.array(musique) / np.max(np.abs(musique)) * 32767)
    
    wavfile.write("beethoven.wav", fe, musique)
    return musique

def create_wav_sound(harmonic, phases, fe, fondamental, enveloppe, duration, filename):
    note = get_sound(harmonic, phases, fe, fondamental, enveloppe, duration)
    note = np.int16(np.array(note) / np.max(np.abs(note)) * 32767)
    wavfile.write("notes/" + filename, fe, note)