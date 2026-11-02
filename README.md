# Parallelizzazione di un Algoritmo per la Generazione di Heatmap

Progetto sviluppato per il corso di **Sistemi di Elaborazione Accelerata** (a.a. 2025/2026).
Puoi consultare la presentazione completa del progetto [qui](docs/Heatmap_Savino_Tolomelli.pdf)

## 📋 Descrizione del Progetto
L'obiettivo del progetto è la generazione efficiente di heatmap a partire da milioni di eventi spaziali (es. dati di tracking calcistico), sfruttando l'accelerazione hardware tramite **GPU (CUDA)** e tecniche diparallelizzazione su **CPU (SIMD / AVX)**.

Il progetto esplora diversi livelli di ottimizzazione:
1. **Implementazione Sequenziale Naïve** (pixel-driven).
2. **Implementazione Parallela Base in CUDA**.
3. **Spatial Partitioning e Shared Memory** per ottimizzare gli accessi alla memoria globale.
4. **Instruction-Level Parallelism (ILP)** e loop unrolling.
5. **Ottimizzazione CPU tramite vettorizzazione SIMD (AVX2 / registri a 256-bit)**.

---

## 🚀 Risultati Principali
* **CUDA:** Sfruttando il partizionamento spaziale e la memoria condivisa, si raggiungono speedup significativi (superiori a $10.000\times$ rispetto al naive)[cite: 10], mantenendo un'occupazione della GPU elevata (fino al ~90%).
* **SIMD (CPU):** L'utilizzo delle istruzioni AVX (`__m256`) permette di elaborare 8 eventi per ciclo, sebbene la CPU mostri limiti fisici evidenti sui dataset massivi rispetto all'architettura parallela massiva della GPU.

---

## 🛠️ Requisiti di Sistema
* **GPU:** NVIDIA Tesla T4 o equivalente (testato su Google Colab / CUDA Toolkit)[cite: 5]
* **CPU:** Supporto istruzioni AVX2
* **Compilatore:** `nvcc` (NVIDIA CUDA Compiler) e `g++`

## ⚙️ Compilazione ed Esecuzione
Esempio di compilazione per il codice CUDA:
```bash
nvcc -O3 src/cuda_kernels.cu -o heatmap_cuda
./heatmap_cuda
```bash
nvcc -O3 src/cuda_kernels.cu -o heatmap_cuda
./heatmap_cuda


