# Engineering High-Throughput Synthetic Data Generation with Local LLMs

> **An empirical evaluation of throughput–quality trade-offs for local LLM inference on consumer GPUs.**

This repository accompanies the technical report **"Engineering High-Throughput Synthetic Data Generation with Local LLMs"**, which benchmarks modern open-weight language models for large-scale synthetic data generation.

The study evaluates how inference engines, batching strategies, concurrency, context allocation, and model architectures affect generation throughput on an **NVIDIA RTX 3050 Laptop GPU (6 GB VRAM)**.

---

## 📄 Technical Report

The complete report is available here:

**➡️ [LLM-Throughput-Technical-Report.pdf](report/LLM-Throughput-Technical-Report.pdf)**

The report contains the complete experimental methodology, benchmark results, analysis, figures, and engineering recommendations.

---

## Highlights

- **41 benchmark configurations**
- **10 state-of-the-art LLMs**
- **2 inference engines**
  - vLLM
  - llama.cpp
- Throughput, latency, VRAM, GPU utilization, and power analysis
- Dense vs. PLE vs. Sparse MoE architecture comparison
- CPU offloading evaluation
- Throughput vs. capability trade-off analysis
- Practical recommendations for local synthetic data generation

---

## Models Evaluated

- Qwen3-0.6B
- Qwen3.5-2B
- Qwen3.5-4B
- Qwen3.5-9B
- Qwen3.6-27B
- Qwen3.6-35B-A3B
- Gemma 4 E2B
- Gemma 4 E4B
- LFM2.5-8B-A1B
- LFM2-24B-A2B

---

## Key Findings

- **vLLM** achieved the highest measured throughput for fully GPU-resident models.
- **PLE architectures** delivered the best throughput–capability trade-off among resident models.
- **Sparse MoE models** substantially outperformed dense models when CPU offloading was required.
- Proper tuning of batching and concurrency had a significant impact on overall throughput.

For detailed experimental results and analysis, please refer to the technical report.
