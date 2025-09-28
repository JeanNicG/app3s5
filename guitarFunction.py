import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

def db_to_linear(db):
    return 10**(db/20)

def N_calc(wc, target_gain):
    N = 0
    gain = 1
    print("target_gain:", target_gain)
    while gain > target_gain:
        N += 1
        gain = np.abs( (1/N) * (np.sin(N*wc/2) / np.sin(wc/2)) ) # 3-51 lyons
    return N

def get_enveloppe(audio_data, filtre):
    enveloppe = np.convolve(np.abs(audio_data), filtre, mode='same')
    return enveloppe / np.max(enveloppe)

def analyse_freq(audio_data, fe):
    # FFT du signal audio
    audio_data_fft = np.fft.fft(audio_data)
    amplitude_audio_data_fft = np.abs(audio_data_fft)
    freq_audio_data_fft = np.fft.fftfreq(len(audio_data), 1/fe)
    
    # Chercher seulement dans les fréquences positives
    n_positive = len(audio_data) // 2
    fondamental_index = np.argmax(amplitude_audio_data_fft[:n_positive])
    fondamental = freq_audio_data_fft[fondamental_index]

    # Convertir les fréquences harmoniques en indices
    harmonic_freqs = [fondamental * i for i in range(1, 33)]
    harmonic_index = [int(round(f * len(audio_data) / fe)) for f in harmonic_freqs]
    
    harmonic = [np.abs(audio_data_fft[i]) for i in harmonic_index]
    phases = [np.angle(audio_data_fft[i]) for i in harmonic_index]
    
    return np.abs(fondamental), harmonic, phases

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
    t = np.linspace(0, duration, int(fe * duration))
    signal_synth = np.zeros(len(t))
    
    for n in range(len(harmonic)):
        freq_harmonic = (n+1) * fondamental
        signal_synth += harmonic[n] * np.sin(2 * np.pi * freq_harmonic * t + phases[n])

    envelope_resampled = signal.resample(enveloppe, len(signal_synth))
    
    signal_synth *= envelope_resampled
    signal_synth *= np.hamming(len(signal_synth))

    return signal_synth

def get_silence(fe, duration):
    t = np.linspace(0, duration, int(fe*duration))
    silence = np.zeros_like(t)
    return silence

def composition_bethoven(harmonic, phases, fe, enveloppe, note_dict):
    sol = get_sound(harmonic, phases, fe, note_dict["SOL"], enveloppe, 0.4)
    mi_b = get_sound(harmonic, phases, fe, note_dict["RE#"], enveloppe, 1)
    fa = get_sound(harmonic, phases, fe, note_dict["FA"], enveloppe, 0.4)
    re = get_sound(harmonic, phases, fe, note_dict["RE"], enveloppe, 1)
    
    silence = get_silence(fe, 0.05)
    silence1 = get_silence(fe, 0.3)

    musique = np.concatenate((
        sol, silence, 
        sol, silence, 
        sol, silence, 
        mi_b, silence1, 
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