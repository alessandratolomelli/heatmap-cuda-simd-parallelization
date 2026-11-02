extern "C" __global__
void compute_heatmap_spatial_ilp(
    float* __restrict__ heatmap,
    const float* __restrict__ x_events,
    const float* __restrict__ y_events,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_ends,
    int grid_w, int grid_h, int width, int height, float cell_size
) {
    // 1. MAPPING PIXEL
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 2. SPATIAL HASHING
    int c_x = (int)((float)x / cell_size);
    int c_y = (int)((float)y / cell_size);
    c_x = max(0, min(c_x, grid_w - 1));
    c_y = max(0, min(c_y, grid_h - 1));
    
    int cell_id = c_y * grid_w + c_x;
    int start = cell_starts[cell_id];
    int end   = cell_ends[cell_id];

    if (start == -1) return;

    int n_events = end - start;
    const int MAX_E = 256; 
    if (n_events > MAX_E) n_events = MAX_E;

    // 3. SHARED MEMORY
    extern __shared__ float shmem[];
    float* sh_x = shmem;
    float* sh_y = shmem + MAX_E;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int n_threads = blockDim.x * blockDim.y;

    for (int i = tid; i < n_events; i += n_threads) {
        sh_x[i] = x_events[start + i];
        sh_y[i] = y_events[start + i];
    }
    __syncthreads();

    // 4. CALCOLO DENSITÀ BINARIA (Senza esponenziale)
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float px = (float)x;
    float py = (float)y;
    
    constexpr float radius_sq = 200.0f;

    for (int i = 0; i < n_events; i += 4) {
        #pragma unroll 4
        for (int k = 0; k < 4; k++) {
            int idx = i + k;
            if (idx < n_events) {
                float dx = px - sh_x[idx];
                float dy = py - sh_y[idx];
                float d2 = dx*dx + dy*dy;
                
                // aggiunta di densità uniforme
                if (d2 < radius_sq) {
                    if (k == 0) a0 += 1.0f;
                    else if (k == 1) a1 += 1.0f;
                    else if (k == 2) a2 += 1.0f;
                    else a3 += 1.0f;
                }
            }
        }
    }

    // 5. SCRITTURA FINALE
    heatmap[y * width + x] = a0 + a1 + a2 + a3;
}