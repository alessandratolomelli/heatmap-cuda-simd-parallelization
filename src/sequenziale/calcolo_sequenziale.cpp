#include <cmath>

extern "C" void compute_heatmap_seq(
    float* heatmap,
    const float* x_events,
    const float* y_events,
    int n_events,
    int width,
    int height)
{
    const float radius_sq = 400.0f;
    const float inv_sigma = 1.0f / 50.0f;

    // loop sui PIXEL (pixel-driven)
    for (int y = 0; y < height; ++y){
        for (int x = 0; x < width; ++x){
            float intensity = 0.0f;
            // accumula contributi di TUTTI gli eventi
            for (int i = 0; i < n_events; ++i){
                float dx = (float)x - x_events[i];
                float dy = (float)y - y_events[i];
                float dist_sq = dx * dx + dy * dy;

                if (dist_sq < radius_sq)
                {
                    intensity += std::exp(-dist_sq * inv_sigma);
                }
            }
            // scrittura finale (una sola volta)
            heatmap[y * width + x] = intensity;
        }
    }
}
