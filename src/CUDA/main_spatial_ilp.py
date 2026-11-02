import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# Percorso per i tuoi moduli personalizzati
sys.path.append("/content/drive/MyDrive/Colab Notebooks/Heatmap")
import campo
import eventi
t_kernel_start = time.perf_counter()  # inizio kernel CPU
# ======================================================
# 0. PARAMETRI GLOBALI
# ======================================================
# Risoluzione della heatmap (es. 10 pixel per metro)
scale = 32  #int Python
w, h = 105 * scale, 68 * scale
cell_size = 20.0 # Dimensione della cella per lo spatial hashing, internamente è un double precision

grid_w = int(np.ceil(w / cell_size))
grid_h = int(np.ceil(h / cell_size))
n_cells = grid_w * grid_h

MAX_EVENTS_PER_CELL = 256

print(f"Dimensioni Heatmap: {w}x{h}")
print(f"Griglia Spatial Hashing: {grid_w}x{grid_h}")

# ======================================================
# 1. GENERAZIONE EVENTI (1 MILIONE)
# ======================================================
np.random.seed(42)
n_totale = 500_000

# Coordinate espresse in metri (0-105, 0-68)
# Cluster centrale (punto d'interesse)
x_focus = np.random.normal(52.5, 10, 50000) #NumPy usa float64 di default
y_focus = np.random.normal(34.0, 10, 50000)

# Rumore uniforme su tutto il campo
x_rand = np.random.uniform(0, 105, 450_000)
y_rand = np.random.uniform(0, 68, 450_000)

x_all = np.concatenate([x_focus, x_rand]).astype(np.float32)
y_all = np.concatenate([y_focus, y_rand]).astype(np.float32)

# Portiamo i dati su GPU
x_gpu = cp.asarray(x_all)
y_gpu = cp.asarray(y_all)

# ======================================================
# 2. SPATIAL HASHING SU GPU
# ======================================================
# Convertiamo i metri in coordinate "pixel" per trovare la cella corretta
cell_x = cp.clip(((x_gpu * scale) / cell_size).astype(cp.int32), 0, grid_w - 1)
cell_y = cp.clip(((y_gpu * scale) / cell_size).astype(cp.int32), 0, grid_h - 1)
cell_ids = cell_y * grid_w + cell_x

# Ordinamento degli eventi per cella
order = cp.argsort(cell_ids)
x_sorted = x_gpu[order]
y_sorted = y_gpu[order]
cell_ids_sorted = cell_ids[order]

# Indici di inizio e fine per ogni cella
cell_starts = cp.full(n_cells, -1, dtype=cp.int32)
cell_ends   = cp.full(n_cells, -1, dtype=cp.int32)

unique_cells, start_idx = cp.unique(cell_ids_sorted, return_index=True)
counts = cp.bincount(cell_ids_sorted, minlength=n_cells)

cell_starts[unique_cells] = start_idx.astype(cp.int32)
cell_ends[unique_cells]   = (start_idx + counts[unique_cells]).astype(cp.int32)

# ======================================================
# 3. CARICAMENTO KERNEL
# ======================================================
with open("/content/drive/MyDrive/Colab Notebooks/Heatmap/CUDA/kernel_spatial_ilp.cu", "r") as f:
    cuda_code = f.read()

kernel = cp.RawKernel(cuda_code, "compute_heatmap_spatial_ilp", options=("--use_fast_math",))

# ======================================================
# 4. PREPARAZIONE OUTPUT E LANCIO
# ======================================================
heatmap_gpu = cp.zeros((h, w), dtype=cp.float32)

threads = (8, 8)
# Copriamo tutti i pixel della heatmap
blocks = (int(np.ceil(w / threads[0])), int(np.ceil(h / threads[1])))

# 256 eventi * 2 coordinate * 4 byte (float) = 2 KB per blocco
shared_mem_bytes = MAX_EVENTS_PER_CELL * 2 * 4 

start_ev = cp.cuda.Event()
end_ev = cp.cuda.Event()
start_ev.record()



kernel(
    blocks, threads,
    (heatmap_gpu, x_sorted * scale, y_sorted * scale, # Passiamo i punti scalati in pixel
     cell_starts, cell_ends, np.int32(grid_w), np.int32(grid_h), 
     np.int32(w), np.int32(h), np.float32(cell_size)),
    shared_mem=shared_mem_bytes
)

end_ev.record()
end_ev.synchronize()
print(f"Tempo Kernel GPU: {cp.cuda.get_elapsed_time(start_ev, end_ev)/1000:.4f} s")
t_kernel_end = time.perf_counter()    # fine kernel CPU

cpu_kernel_time = t_kernel_end - t_kernel_start
print(f"Tempo CPU kernel: {cpu_kernel_time:.4f} s")
# ======================================================
# 5. VISUALIZZAZIONE
# ======================================================
heatmap_cpu = cp.asnumpy(heatmap_gpu)

plt.figure(figsize=(12, 8))
ax = plt.gca()

# L'extent deve corrispondere ai metri del campo per il disegno corretto
im = ax.imshow(heatmap_cpu, extent=[0, 105, 0, 68], origin='lower', 
               cmap='Greens', alpha=0.8, interpolation='bilinear')

campo.crea_campo_105x68(ax)
plt.title("Heatmap Ottimizzata (CUDA Spatial Hashing + ILP)")
plt.show()