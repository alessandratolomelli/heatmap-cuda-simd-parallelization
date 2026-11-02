import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/content/drive/MyDrive/Colab Notebooks/Heatmap")
import campo
import eventi
import time

t_end_to_end_start = time.perf_counter()

# ======================================================
# 0. PARAMETRI GLOBALI
# ======================================================
w, h = 105 * 128, 68 * 128  
cell_size = 20.0
grid_w = int(np.ceil(w / cell_size))
grid_h = int(np.ceil(h / cell_size))        # dimensioni heatmap

n_cells = grid_w * grid_h

MAX_EVENTS_PER_CELL = 256          # limite massimo shared memory

print("Grid celle:", grid_w, "x", grid_h)

# ======================================================
# 1. GENERAZIONE EVENTI (1 MILIONE)
# ======================================================
np.random.seed(0)

centro_x, centro_y = 52.5, 34
dev = 10

# cluster centrale
x_focus = np.random.normal(centro_x, dev, 130_000)
y_focus = np.random.normal(centro_y, dev, 130_000)

# rumore uniforme
x_rand = np.random.uniform(0, w, 870_000)
y_rand = np.random.uniform(0, h, 870_000)

x_all = np.concatenate([x_focus, x_rand]).astype(np.float32)
y_all = np.concatenate([y_focus, y_rand]).astype(np.float32)

x_gpu = cp.asarray(x_all)
y_gpu = cp.asarray(y_all)

#2



# assegnazione eventi alle celle
cell_x = cp.clip((x_gpu / cell_size).astype(cp.int32), 0, grid_w - 1)
cell_y = cp.clip((y_gpu / cell_size).astype(cp.int32), 0, grid_h - 1)
cell_ids = cell_y * grid_w + cell_x

# Ordina eventi per cella
order = cp.argsort(cell_ids)
x_sorted = x_gpu[order]
y_sorted = y_gpu[order]
cell_ids_sorted = cell_ids[order]

# Inizio / fine eventi per cella
cell_starts = cp.full(n_cells, -1, dtype=cp.int32)
cell_ends   = cp.full(n_cells, -1, dtype=cp.int32)

unique_cells, start_idx = cp.unique(cell_ids_sorted, return_index=True)
counts = cp.bincount(cell_ids_sorted, minlength=n_cells)

cell_starts[unique_cells] = start_idx.astype(cp.int32)
cell_ends[unique_cells]   = (start_idx + counts[unique_cells]).astype(cp.int32)

# ======================================================
# 3. CARICAMENTO KERNEL CUDA
# ======================================================
with open("/content/drive/MyDrive/Colab Notebooks/Heatmap/CUDA/kernel_spatial.cu", "r") as f:
    cuda_code = f.read()

kernel = cp.RawKernel(
    cuda_code,
    "compute_heatmap_spatial",
    options=(
        "--use_fast_math",
        "--maxrregcount=64",
    )
)

# ======================================================
# 4. HEATMAP
# ======================================================
heatmap_gpu = cp.zeros((h, w), dtype=cp.float32)

# ======================================================
# 5. CONFIGURAZIONE LAUNCH
# ======================================================
threads = (16,16)
blocks = (grid_w, grid_h)

shared_mem_bytes = MAX_EVENTS_PER_CELL * 2 * 4  # ogni evento ha due coordinate con peso 4B

print("Threads per block:", threads)
print("Shared memory per block:", shared_mem_bytes / 1024, "KB")

# ======================================================
# 6. ESECUZIONE + TIMING GPU
# ======================================================
start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()

kernel(
    blocks,
    threads,
    (
        heatmap_gpu,
        x_sorted,
        y_sorted,
        cell_starts,
        cell_ends,
        np.int32(grid_w),
        np.int32(grid_h),
        np.int32(w),
        np.int32(h),
        np.float32(cell_size),
    ),
    shared_mem=shared_mem_bytes
)

end.record()
end.synchronize()

print(f"Tempo GPU: {cp.cuda.get_elapsed_time(start, end) / 1000:.4f} s")
t_end_to_end_end = time.perf_counter()
print(f"Tempo END-to-END totale: {t_end_to_end_end - t_end_to_end_start:.4f} s")

# ======================================================
# 7. VISUALIZZAZIONE
# ======================================================
plt.figure(figsize=(12, 8))
ax = plt.gca()

heatmap_cpu = cp.asnumpy(heatmap_gpu)
im = ax.imshow(
    heatmap_cpu,
    extent=[0, w, 0, h],
    origin='lower',
    cmap='Greens',
    alpha=0.7,
    interpolation='nearest'
)

# Griglia nera
for x in range(w + 1):
    ax.axvline(x, color='black', lw=0.3, alpha=0.3, zorder=6)
for y in range(h + 1):
    ax.axhline(y, color='black', lw=0.3, alpha=0.3, zorder=6)

# Campo 105x68
campo.crea_campo_105x68(ax)
plt.show()