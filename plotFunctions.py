import matplotlib.pyplot as plt


plt.figure()
plt.plot(enveloppe)
plt.title("Enveloppe temporelle du signal audio")
plt.xlabel("Echantillons")
plt.ylabel("Amplitude normalisée")
plt.grid()
plt.figure()
plt.plot(amplitude_db)
plt.title("Réponse en fréquence du filtre passe-bas")
plt.xlabel("Fréquence (rad/échantillon)")
plt.ylabel("Amplitude (dB)")
plt.grid()

plt.figure()
plt.stem(range(len(h_shift)), h_shift)
plt.title("Réponse impulsionnelle du filtre passe-bas")
plt.xlabel("Echantillons")
plt.ylabel("Amplitude")
plt.grid()
plt.show()