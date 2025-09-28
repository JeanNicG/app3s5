import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import guitarFunction as sf

fe, audio_data = wavfile.read('note_guitare_lad.wav')

# 1.Obtention de l'envelope temporelle en redressant le signal temporel
audio_data_abs = np.abs(audio_data)

# 2.Conception d'un filtre passe-bas RIF
target_gain_db = -3
target_gain = sf.db_to_linear(target_gain_db)
wc  = np.pi / 1000
N = sf.N_calc(wc , target_gain)
print("Ordre N du filtre:", N)
filtre = np.ones(N+1) / (N+1)
enveloppe = sf.get_enveloppe(audio_data, filtre)

audio_data = audio_data / np.max(audio_data)

fondamental, harmonic, phases = sf.analyse_freq(audio_data, fe)
print("Fréquence fondamentale:", fondamental)

note_dict = sf.note_dict(fondamental)

#sf.create_wav_sound(harmonic, phases, fe, note_dict["SOL"], enveloppe, 4, "sol.wav")
#sf.create_wav_sound(harmonic, phases, fe, note_dict["RE#"], enveloppe, 4, "mi.wav")
#sf.create_wav_sound(harmonic, phases, fe, note_dict["FA"], enveloppe, 4, "fa.wav")
#sf.create_wav_sound(harmonic, phases, fe, note_dict["RE"], enveloppe, 4, "re.wav")

sf.composition_bethoven(harmonic, phases, fe, enveloppe, note_dict)





