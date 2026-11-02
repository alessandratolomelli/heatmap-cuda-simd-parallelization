extern "C" __global__
void compute_heatmap_spatial(
    float* __restrict__ heatmap,
    const float* __restrict__ x_events,
    const float* __restrict__ y_events,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_ends,
    int grid_w,
    int grid_h,
    int width,
    int height,
    float cell_size
) {
    // ---------------------------------
    // Identificazione cella
    // ---------------------------------
    int cell_x = blockIdx.x;
    int cell_y = blockIdx.y;

    if (cell_x >= grid_w || cell_y >= grid_h) return;

    int cell_id = cell_y * grid_w + cell_x;
    int start = cell_starts[cell_id];
    int end   = cell_ends[cell_id];

    if (start == -1) return;

    int n_events = end - start;
    const int MAX_EVENTS = 256;
    n_events = min(n_events, MAX_EVENTS);

    // ---------------------------------
    // Shared memory
    // ---------------------------------
    extern __shared__ float shmem[];
    float* sh_x = shmem;
    float* sh_y = shmem + MAX_EVENTS;

    // Caricamento cooperativo
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int n_threads = blockDim.x * blockDim.y;

    for (int i = tid; i < n_events; i += n_threads) {
        sh_x[i] = x_events[start + i];
        sh_y[i] = y_events[start + i];
    }

    __syncthreads();

    // ---------------------------------
    // Coordinate pixel (CORRETTE)
    // ---------------------------------
    int x = int(cell_x * cell_size) + threadIdx.x;
    int y = int(cell_y * cell_size) + threadIdx.y;

    if (x >= width || y >= height) return;

    // ---------------------------------
    // Costanti
    // ---------------------------------
    constexpr float radius_sq = 400.0f;
    constexpr float inv_sigma = 1.0f / 50.0f;

    float intensity = 0.0f;

    // ---------------------------------
    // Loop eventi (shared memory)
    // ---------------------------------
    #pragma unroll 4
    for (int i = 0; i < n_events; ++i) {
        float dx = x - sh_x[i];
        float dy = y - sh_y[i];
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < radius_sq) {
            intensity += __expf(-dist_sq * inv_sigma);
        }
    }

    heatmap[y * width + x] = intensity;
}
