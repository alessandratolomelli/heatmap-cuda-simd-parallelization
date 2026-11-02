import cupy as cp
import sys
sys.path.append("/content/drive/MyDrive/Colab Notebooks/Heatmap")
import matplotlib.pyplot as plt
import eventi
import numpy as np
import campo
import importlib
importlib.reload(campo)
importlib.reload(eventi)
import time

# ==============================
# 0. Timing END-to-END
# ==============================
t_end_to_end_start = time.perf_counter()

# ==============================
# 1. Preparazione Eventi
# ==============================
centro_x, centro_y = 52.5, 34
deviazione = 10

# Genera punti intorno al centro
x_intorno = np.random.normal(centro_x, deviazione, 130000)
y_intorno = np.random.normal(centro_y, deviazione, 130000)
posizioni_manuali = list(zip(x_intorno, y_intorno))

dati_partita = eventi.genera_eventi(
    n_totale=1000000,
    manuali=posizioni_manuali
)

# Copia in GPU
x_gpu = cp.asarray(dati_partita[:, 0], dtype=cp.float32)
y_gpu = cp.asarray(dati_partita[:, 1], dtype=cp.float32)

# ==============================
# 2. Dimensioni heatmap
# ==============================
w, h = 105*2, 68*2 # dimensioni campo
heatmap_gpu = cp.zeros((h, w), dtype=cp.float32)

# ==============================
# 3. Caricamento kernel pixel-driven
# ==============================
with open('/content/drive/MyDrive/Colab Notebooks/Heatmap/CUDA/kernel_standard.cu', 'r') as f:
    cuda_code = f.read()

heatmap_kernel = cp.RawKernel(cuda_code, 'compute_heatmap_standard')

# ==============================
# 4. Dimensioni blocchi / griglia
# ==============================
threads = (16, 16)
blocks = ((w + threads[0] - 1) // threads[0],
          (h + threads[1] - 1) // threads[1])

# ==============================
# 5. Esecuzione + Timing GPU
# ==============================
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()

start_event.record()

heatmap_kernel(
    blocks,
    threads,
    (
        heatmap_gpu,
        x_gpu,
        y_gpu,
        np.int32(len(x_gpu)),
        np.int32(w),
        np.int32(h)
    )
)

end_event.record()
end_event.synchronize()
tempo_ms = cp.cuda.get_elapsed_time(start_event, end_event)
print(f"Tempo GPU preciso: {tempo_ms / 1000:.6f} secondi")

# ==============================
# 6. Timing END-to-END
# ==============================
t_end_to_end_end = time.perf_counter()
print(f"Tempo END-to-END totale: {t_end_to_end_end - t_end_to_end_start:.4f} s")

# ==============================
# 7. Visualizzazione
# ==============================
plt.figure(figsize=(12, 8))
ax = plt.gca()

heatmap_cpu = cp.asnumpy(heatmap_gpu)
im = ax.imshow(
    heatmap_cpu,
    extent=[0, w, 0, h],
    origin='lower',
    cmap='Greens',
    alpha=0.9,
    interpolation='bilinear'
)

# Griglia nera
for x in range(w + 1):
    ax.axvline(x, color='black', lw=0.3, alpha=0.3, zorder=6)
for y in range(h + 1):
    ax.axhline(y, color='black', lw=0.3, alpha=0.3, zorder=6)

# Campo 105x68
campo.crea_campo_105x68(ax)
plt.show()
