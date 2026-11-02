# Parallelization of an Algorithm for Heatmap Generation

Project developed for the course **Accelerated Processing Systems** (A.Y. 2025/2026).
You can consult the complete project presentation [here](docs/Heatmap_Savino_Tolomelli.pdf).

## 📋 Project Description
The objective of the project is the efficient generation of heatmaps starting from millions of spatial events (e.g., soccer tracking data), leveraging hardware acceleration via **GPU (CUDA)** and parallelization techniques on **CPU (SIMD / AVX)**.

The project explores several levels of optimization:
1. **Naive Sequential Implementation** (pixel-driven).
2. **Basic Parallel Implementation in CUDA**.
3. **Spatial Partitioning and Shared Memory** to optimize global memory accesses.
4. **Instruction-Level Parallelism (ILP)** and loop unrolling.
5. **CPU Optimization via SIMD Vectorization (AVX2 / 256-bit registers)**.

---

## 🚀 Main Results
- **CUDA:** By leveraging spatial partitioning and shared memory, significant speedups are achieved (exceeding $10,000\times$ compared to the naive approach), while maintaining high GPU occupancy (up to ~90%).
- **SIMD (CPU):** The use of AVX instructions (`__m256`) allows processing 8 events per cycle, although the CPU shows clear physical limits on massive datasets compared to the massively parallel architecture of the GPU.

---

## 🛠️ System Requirements
- **GPU:** NVIDIA Tesla T4 or equivalent (tested on Google Colab / CUDA Toolkit)
- **CPU:** AVX2 instruction support
- **Compiler:** `nvcc` (NVIDIA CUDA Compiler) and `g++`
