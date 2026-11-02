extern "C" __global__
void compute_heatmap_standard(
    float* heatmap,
    const float* x_events,
    const float* y_events,
    int n_events,
    int width,
    int height)
{
    const float radius_sq = 400.0f;   // raggio^2
    const float inv_sigma = 1.0f / 50.0f;

    // calcolo coordinate del pixel assegnato a questo thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float intensity = 0.0f;

    // loop su tutti gli eventi
    for (int i = 0; i < n_events; ++i)
    {
        float dx = (float)x - x_events[i];
        float dy = (float)y - y_events[i];
        float dist_sq = dx*dx + dy*dy;

        if (dist_sq < radius_sq)
            intensity += expf(-dist_sq * inv_sigma);
    }

    // scrittura finale del pixel
    heatmap[y * width + x] = intensity;
}
