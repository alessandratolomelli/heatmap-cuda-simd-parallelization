import ctypes
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/content/drive/MyDrive/Colab Notebooks/Heatmap")
import campo   # Il tuo file grafico
import eventi  # Il tuo generatore di punti
import time

# ================================
# MISURA END-TO-END (inizio)
# ================================
t_end_to_end_start = time.perf_counter()


# Carichiamo la libreria C++ che abbiamo appena compilato
libreria = ctypes.CDLL("/content/drive/MyDrive/Colab Notebooks/Heatmap/SIMD/calcolo_simd.so")

n_punti_centro = 130000  #punti forzati
centro_x, centro_y = 52.5, 34.0
dispersione = 12   # Più è basso, più sono vicini al punto centrale

# Generiamo coordinate X e Y concentrate
x_centro = np.random.normal(centro_x, dispersione, n_punti_centro)
y_centro = np.random.normal(centro_y, dispersione, n_punti_centro)

# Uniamo le coordinate in una lista di tuple (x, y)
punti_forzati = list(zip(x_centro, y_centro))

# --- GENERAZIONE FINALE ---
# Usiamo la tua funzione eventi.genera_eventi passandogli i punti forzati
# n_totale può essere 1000: 500 saranno al centro, 500 casuali altrove
dati = eventi.genera_eventi(n_totale=1000000, manuali=punti_forzati)
# Prepariamo i dati
w, h = 105*4, 68*4
x_punti = dati[:, 0].astype(np.float32)
y_punti = dati[:, 1].astype(np.float32)
matrice_risultato = np.zeros((h, w), dtype=np.float32)

# Chiamata al C++ (misurando il tempo)
t_kernel_start = time.perf_counter()
libreria.compute_heatmap_simd(
    matrice_risultato.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    x_punti.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    y_punti.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    len(dati), w, h
)

print(f"Tempo C++ SIMD: {time.perf_counter() - t_kernel_start:.4f} secondi")

t_kernel_end = time.perf_counter()

# ================================
# MISURA END-TO-END (fine)
# ================================
t_end_to_end_end = time.perf_counter()

# -------------------------------
# Stampa tempi
# -------------------------------
print(f"Tempo KERNEL C++ (solo calcolo): "
      f"{t_kernel_end - t_kernel_start:.4f} s")

print(f"Tempo END-TO-END Sequenziale (totale applicazione): "
      f"{t_end_to_end_end - t_end_to_end_start:.4f} s")

# --- 4. Statistiche dati ---
print(f"🔢 Numero eventi: {len(x_punti)}")
print(f"🔢 Media X: {x_punti.mean():.2f}, Media Y: {y_punti.mean():.2f}")
print(f"🔢 Std X: {x_punti.std():.2f}, Std Y: {y_punti.std():.2f}") #quanto i valori degli eventi si disperdono intorno alla media.
print(f"🔢 Min/Max X: {x_punti.min():.2f}/{x_punti.max():.2f}")
print(f"🔢 Min/Max Y: {y_punti.min():.2f}/{y_punti.max():.2f}")
# Visualizzazione con il tuo campo.py
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(matrice_risultato, extent=[0, w, 0, h], origin='lower', 
               cmap='Greens', alpha=0.8, interpolation='bilinear')


               # --- AGGIUNTA GRIGLIA NERA ---
# Disegniamo le linee verticali (ogni metro lungo la X)
for x in range(w+1):
    ax.axvline(x, color='black', lw=0.3, alpha=0.3, zorder=6)

# Disegniamo le linee orizzontali (ogni metro lungo la Y)
for y in range(h+1):
    ax.axhline(y, color='black', lw=0.3, alpha=0.3, zorder=6)

campo.crea_campo_105x68(ax) # <-- Ecco il tuo file!
plt.show()