import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create output folder
os.makedirs('./plots', exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11.5,
    'axes.labelsize': 13.5,
    'axes.titlesize': 15.0,
    'legend.fontsize': 11.5,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 11.5
})

# 1. vLLM Concurrency vs Throughput
plt.figure(figsize=(8, 4.5))
x_2b = [8, 16, 32, 64, 113]
y_2b = [504.74, 829.39, 1097.26, 1149.07, 1127.19]
x_06b = [64, 128, 160]
y_06b = [2532.42, 3160.42, 2879.63]
plt.plot(x_2b, y_2b, marker='o', label='Qwen 3.5 2B AWQ', color='#2B4C7E', linewidth=2)
plt.plot(x_06b, y_06b, marker='s', label='Qwen 3 0.6B GPTQ', color='#E05D5D', linewidth=2)
plt.xlabel("Concurrency (Parallel Requests)")
plt.ylabel("Completion Throughput (tokens/sec)")
plt.title("vLLM: Concurrency vs Throughput")
plt.legend()
plt.tight_layout()
plt.savefig('./plots/vllm_concurrency_vs_throughput.png', dpi=300)
plt.close()

# 2. vLLM Model Size vs Throughput
plt.figure(figsize=(6, 4))
sizes = ['Qwen 3.5 2B (AWQ)', 'Qwen 3 0.6B (GPTQ)']
speeds = [1149.07, 3160.42]
colors = ['#2B4C7E', '#E05D5D']
plt.bar(sizes, speeds, color=colors, width=0.4)
plt.ylabel("Completion Throughput (tokens/sec)")
plt.title("vLLM: Max Throughput vs Model Scale")
plt.tight_layout()
plt.savefig('./plots/vllm_size_vs_throughput.png', dpi=150)
plt.close()

# 3. Batch Size vs Throughput, Latency, Util
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
batches = ['2048', '4096', '8192']
throughput = [150.01, 178.34, 162.19]
latency = [46.055, 38.736, 42.601]

ax1.bar(batches, throughput, color='#2B4C7E', width=0.4)
ax1.set_xlabel("Prefill Batch Size (--batch-size)")
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_title("Throughput vs Prefill Batch Size")

ax2.plot(batches, latency, marker='o', color='#E05D5D', linewidth=2)
ax2.set_xlabel("Prefill Batch Size (--batch-size)")
ax2.set_ylabel("Mean Latency (s)")
ax2.set_title("Latency vs Prefill Batch Size")
plt.tight_layout()
plt.savefig('./plots/batch_size_metrics.png', dpi=150)
plt.close()

# 4. Micro-Batch vs Throughput, Power, Util
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ubatch = ['256', '512', '1024']
throughput_u = [144.49, 178.34, 119.03]
power_u = [66.6, 67.3, 63.5]

ax1.bar(ubatch, throughput_u, color='#3B6C9E', width=0.4)
ax1.set_xlabel("Micro-prefill ubatch (--ubatch-size)")
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_title("Throughput vs Micro-prefill Size")

ax2.plot(ubatch, power_u, marker='s', color='#E05D5D', linewidth=2)
ax2.set_xlabel("Micro-prefill ubatch (--ubatch-size)")
ax2.set_ylabel("Average GPU Power (W)")
ax2.set_title("GPU Power Draw vs Micro-prefill Size")
plt.tight_layout()
plt.savefig('./plots/ubatch_metrics.png', dpi=150)
plt.close()

# 5. Concurrency Slots vs Throughput, Latency, VRAM
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
slots = [62, 120, 150, 192, 256]
tp = [399.76, 442.47, 496.13, 502.17, 484.34]
vram = [5.76, 5.73, 5.67, 5.50, 5.16]

ax1.plot(slots, tp, marker='o', color='#2B4C7E', linewidth=2)
ax1.set_xlabel("Concurrency Slots (--parallel)")
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_title("Throughput vs Concurrency slots")

ax2.plot(slots, vram, marker='d', color='#2CA02C', linewidth=2)
ax2.set_xlabel("Concurrency Slots (--parallel)")
ax2.set_ylabel("VRAM Usage (GB)")
ax2.set_title("VRAM Allocation vs Concurrency slots")
plt.tight_layout()
plt.savefig('./plots/slots_metrics.png', dpi=150)
plt.close()

# 6. Context Size vs Throughput and VRAM
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ctx = ['131,072', '368,640']
ctx_tp = [926.56, 908.18]
ctx_vram = [3.928, 5.466]

ax1.bar(ctx, ctx_tp, color='#2B4C7E', width=0.3)
ax1.set_xlabel("Context Size (--ctx-size)")
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_title("Throughput vs Context Size")

ax2.bar(ctx, ctx_vram, color='#FF7F0E', width=0.3)
ax2.set_xlabel("Context Size (--ctx-size)")
ax2.set_ylabel("Max VRAM Allocation (GB)")
ax2.set_title("VRAM Allocation vs Context Size")
plt.tight_layout()
plt.savefig('./plots/context_metrics.png', dpi=150)
plt.close()

# 7. Architecture Comparison (Throughput vs Active Params)
plt.figure(figsize=(7, 4))
models = ['LFM 8B (MoE)', 'Gemma4 E2B (PLE)', 'Qwen3.5 4B (Dense)', 'Gemma4 E4B (PLE)']
active_params = [1.0, 2.3, 4.0, 4.5]
tps = [442.47, 926.56, 178.34, 503.90]

plt.scatter(active_params, tps, s=150, color='#2B4C7E', alpha=0.8)
for i, txt in enumerate(models):
    plt.annotate(txt, (active_params[i], tps[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel("Active Parameter Size per Token (B)")
plt.ylabel("Measured Throughput (tokens/sec)")
plt.title("Performance efficiency by Active Parameter Count")
plt.xlim(0, 5.5)
plt.ylim(0, 1100)
plt.tight_layout()
plt.savefig('./plots/architecture_active_params.png', dpi=150)
plt.close()

# 8. Throughput vs Model
plt.figure(figsize=(7, 4))
models_all = ['Qwen3-0.6B', 'Qwen3.5-2B', 'Gemma4-E2B', 'Gemma4-E4B', 'LFM2.5-8B', 'Qwen3.5-4B', 'Qwen3.5-9B', 'LFM2-24B']
tps_all = [3160.42, 1149.07, 926.56, 503.90, 502.17, 178.34, 153.38, 51.11]
plt.bar(models_all, tps_all, color='#2B4C7E', width=0.5)
plt.xticks(rotation=45)
plt.ylabel("Completion Throughput (tok/s)")
plt.title("Measured Throughput Across All Fully Resident Models")
plt.tight_layout()
plt.savefig('./plots/all_models_throughput.png', dpi=150)
plt.close()

# 9. CPU Offloading: Concurrency vs Throughput & Latency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
concur = [1, 2, 4, 8]
tp_off = [2.56, 3.97, 4.01, 1.35]
lat_off = [19.437, 25.154, 24.933, 147.579]

ax1.plot(concur, tp_off, marker='o', color='#E05D5D', linewidth=2)
ax1.set_xlabel("Client Concurrency (Slots)")
ax1.set_ylabel("Throughput (tok/s)")
ax1.set_title("CPU Offloading: Concurrency vs Throughput")

ax2.plot(concur, lat_off, marker='s', color='#2B4C7E', linewidth=2)
ax2.set_xlabel("Client Concurrency (Slots)")
ax2.set_ylabel("Mean Latency (s)")
ax2.set_title("CPU Offloading: Concurrency vs Latency")
plt.tight_layout()
plt.savefig('./plots/cpu_offloading_metrics.png', dpi=150)
plt.close()

# 10. Speed vs Intelligence: Quality vs Throughput (Pareto Frontier)
plt.figure(figsize=(8, 4.5))
models_spec = ['Qwen3.5-2B', 'Gemma4-E2B', 'Gemma4-E4B', 'LFM2.5-8B', 'Qwen3.5-4B', 'Qwen3.5-9B', 'LFM2-24B', 'Qwen3.6-35B', 'Qwen3.6-27B']
cii = [48.95, 46.75, 63.48, 57.25, 72.05, 77.15, 55.38, 84.58, 83.47]
tps_spec = [1149.07, 926.56, 503.90, 502.17, 178.34, 153.38, 51.11, 11.00, 4.01]

# Scatter points with larger size
plt.scatter(tps_spec, cii, s=180, color='#2B4C7E', alpha=0.85, zorder=5, label='Evaluated Models')

# Pareto frontier line: (Qwen3.5-2B, Gemma4-E4B, Qwen3.5-9B, Qwen3.6-35B)
tps_pareto = [1149.07, 503.90, 153.38, 11.00]
cii_pareto = [48.95, 63.48, 77.15, 84.58]
plt.plot(tps_pareto, cii_pareto, color='#E05D5D', linestyle='--', linewidth=2.0, alpha=0.9, zorder=3, label='Pareto Frontier')

# Annotations dictionary with offsets to prevent overlap
offsets = {
    'Qwen3.5-2B': (0, 12),
    'Gemma4-E2B': (-32, -18),
    'Gemma4-E4B': (0, 12),
    'LFM2.5-8B': (-34, -18),
    'Qwen3.5-4B': (22, -18),
    'Qwen3.5-9B': (0, 12),
    'LFM2-24B': (32, 10),
    'Qwen3.6-27B': (-34, -18),
    'Qwen3.6-35B': (22, 12)
}

for i, txt in enumerate(models_spec):
    off = offsets.get(txt, (0, 12))
    plt.annotate(txt, (tps_spec[i], cii[i]), textcoords="offset points", xytext=off, ha='center', fontsize=9.5, fontweight='semibold')

plt.xscale('log')
plt.xlabel("Measured Throughput (tokens/sec, Log Scale)", fontweight='bold', labelpad=8)
plt.ylabel("Composite Intelligence Index (%)", fontweight='bold', labelpad=8)
plt.title("The Speed ↔ Intelligence Spectrum (RTX 3050)", fontweight='bold', pad=15)
plt.xlim(2, 2000)
plt.ylim(25, 95)

# Style axes and grid
ax = plt.gca()
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_linewidth(1.2)
    ax.spines[spine].set_color('#444444')

plt.legend(loc='lower left', frameon=True, framealpha=0.9, facecolor='white', edgecolor='#cccccc')
plt.tight_layout()
plt.savefig('./plots/speed_vs_intelligence.png', dpi=300)
plt.close()

# 11. Dataset Size vs Generation Days (Log scale horizontal bar chart)
plt.figure(figsize=(8, 4.5))
models_days = ['Qwen3-0.6B', 'Gemma4-E2B', 'Gemma4-E4B', 'LFM2.5-8B', 'Qwen3.5-9B', 'Qwen3.6-35B', 'Qwen3.6-27B']
days_100m = [0.37, 1.25, 2.30, 2.31, 7.55, 105.22, 288.63]
colors_days = ['#2B4C7E', '#3B6C9E', '#4B8CBE', '#5BACE0', '#FF7F0E', '#E05D5D', '#C82333']

plt.barh(models_days[::-1], days_100m[::-1], color=colors_days[::-1], height=0.6)
plt.xscale('log')
plt.xlabel("Days to Generate 100 Million Tokens (Log Scale)")
plt.title("Pipeline Efficiency: Time Required to Generate 100 Million Tokens")
plt.axvline(x=1.0, color='#888888', linestyle=':', alpha=0.6, label='1 Day')
plt.axvline(x=2.0, color='#aaaaaa', linestyle=':', alpha=0.6, label='2 Days')
plt.axvline(x=7.0, color='#666666', linestyle='--', alpha=0.6, label='1 Week')
plt.axvline(x=30.0, color='#444444', linestyle='--', alpha=0.7, label='1 Month')
plt.axvline(x=365.0, color='#cc0000', linestyle='--', alpha=0.7, label='1 Year')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/dataset_size_vs_days.png', dpi=300)
plt.close()

print("All 11 visualization plots generated successfully inside './plots/'.")

