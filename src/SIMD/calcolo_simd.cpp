#include <immintrin.h>
#include <cmath>

extern "C" void compute_heatmap_simd(
    float* heatmap,
    const float* x_events,
    const float* y_events,
    int n_events,
    int width,
    int height)
{
    const float radius_sq = 400.0f;
    const float inv_sigma = 1.0f / 50.0f;

    const __m256 v_radius = _mm256_set1_ps(radius_sq);

    // loop sui pixel (pixel-driven)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float intensity = 0.0f;

            const __m256 vx = _mm256_set1_ps((float)x);
            const __m256 vy = _mm256_set1_ps((float)y);

            int i = 0;

            // 8 eventi alla volta
            for (; i <= n_events - 8; i += 8)
            {
                __m256 ex = _mm256_loadu_ps(&x_events[i]);
                __m256 ey = _mm256_loadu_ps(&y_events[i]);

                __m256 dx = _mm256_sub_ps(vx, ex);
                __m256 dy = _mm256_sub_ps(vy, ey);

                __m256 dist_sq =
                    _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx));

                // maschera: dist_sq < radius_sq
                __m256 mask = _mm256_cmp_ps(dist_sq, v_radius, _CMP_LT_OQ);

                // scarichiamo solo i valori validi
                float dist_arr[8];
                int mask_bits = _mm256_movemask_ps(mask);

                if (mask_bits != 0)
                {
                    _mm256_storeu_ps(dist_arr, dist_sq);

                    #pragma unroll
                    for (int k = 0; k < 8; ++k)
                    {
                        if (mask_bits & (1 << k))
                        {
                            intensity += __expf(-dist_arr[k] * inv_sigma);
                        }
                    }
                }
            }

            // coda scalare
            for (; i < n_events; ++i)
            {
                float dx = (float)x - x_events[i];
                float dy = (float)y - y_events[i];
                float dist_sq = dx * dx + dy * dy;

                if (dist_sq < radius_sq)
                {
                    intensity += std::exp(-dist_sq * inv_sigma);
                }
            }

            // scrittura finale
            heatmap[y * width + x] = intensity;
        }
    }
}
