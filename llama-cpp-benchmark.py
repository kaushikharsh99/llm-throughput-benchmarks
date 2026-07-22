import asyncio
import json
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional
import aiohttp
# CONFIG - edit these values directly, then just run: python llamacpp_benchmark.py
URL             = "http://127.0.0.1:8080/v1/chat/completions"                         
MODEL           = "LFM2-24B-A2B-Q4_K_M"                                                              
NUM_REQUESTS    = 128                                      
CONCURRENCY     = 128                                                                                
MAX_TOKENS      = 25                              
TEMPERATURE     = 0.7
PROMPT          = None
PROMPT_FILE     = "prompt.txt"                                                                                                     
CHAT            = True                                                                    
STREAM          = False                                                      
REQUEST_TIMEOUT = 1200.0                                    
JSON_OUT        = None                                                                             
LLAMA_QUANT     = "Q4_K_M"
LLAMA_GPU_LAYERS = 14                                         
LLAMA_CTX_SIZE  = 65536                 
LLAMA_SLOTS     = 128                                                   
LLAMA_BATCH     = 4096                    
LLAMA_UBATCH    = 512                      
MONITOR_GPU     = True                                                     
MONITOR_INTERVAL = 1.0                               
@dataclass

class RequestResult:
    idx: int
    success: bool
    start_time: float
    end_time: float
    ttft: Optional[float] = None                                                
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    @property
    def latency(self) -> float:
        return self.end_time - self.start_time
    @property
    def tokens_per_sec(self) -> float:
        return self.completion_tokens / self.latency if self.latency > 0 else 0.0
@dataclass

class GpuSample:
    util_pct: float
    mem_used_mb: float
    power_w: float
    temp_c: float

def build_payload(args, prompt: str, use_stream: bool) -> dict:
    if args.chat:
        return {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": use_stream,
        }
    else:
        return {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": use_stream,
        }
async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    idx: int,
    stream: bool,
    timeout_s: float,
) -> RequestResult:
    start = time.perf_counter()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    idx=idx, success=False, start_time=start,
                    end_time=time.perf_counter(),
                    error=f"HTTP {resp.status}: {body[:200]}",
                )
            if stream:
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choice = chunk.get("choices", [{}])[0]
                    if "text" in choice and choice["text"]:
                        completion_tokens += 1                                         
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        completion_tokens += 1
                    usage = chunk.get("usage")
                    if usage:
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            else:
                data = await resp.json()
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                timings = data.get("timings")
                if timings and ttft is None:
                    ttft = timings.get("prompt_ms", 0) / 1000.0 if "prompt_ms" in timings else None
        end = time.perf_counter()
        return RequestResult(
            idx=idx, success=True, start_time=start, end_time=end,
            ttft=ttft, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except asyncio.TimeoutError:
        return RequestResult(
            idx=idx, success=False, start_time=start,
            end_time=time.perf_counter(), error="timeout",
        )
    except Exception as e:
        return RequestResult(
            idx=idx, success=False, start_time=start,
            end_time=time.perf_counter(), error=str(e),
        )
async def worker(sem: asyncio.Semaphore, *args, **kwargs) -> RequestResult:
    async with sem:
        return await send_request(*args, **kwargs)

def load_prompts(args) -> List[str]:
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            raise ValueError("prompt file is empty")
        return [prompts[i % len(prompts)] for i in range(args.num_requests)]
    else:
        return [args.prompt] * args.num_requests

def _nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None

def sample_gpu() -> Optional[GpuSample]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=2.0,
        ).decode("utf-8").strip()
        first_line = out.splitlines()[0]
        util, mem, power, temp = [x.strip() for x in first_line.split(",")]
        return GpuSample(
            util_pct=float(util),
            mem_used_mb=float(mem),
            power_w=float(power) if power not in ("", "N/A") else 0.0,
            temp_c=float(temp),
        )
    except Exception:
        return None
async def gpu_monitor_loop(interval: float, samples: List[GpuSample], stop_event: asyncio.Event):
    while not stop_event.is_set():
        s = sample_gpu()
        if s is not None:
            samples.append(s)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
async def run_benchmark(args):
    prompts = load_prompts(args)
    sem = asyncio.Semaphore(args.concurrency)
    gpu_samples: List[GpuSample] = []
    stop_event = asyncio.Event()
    monitor_task = None
    if args.monitor_gpu and _nvidia_smi_available():
        monitor_task = asyncio.create_task(
            gpu_monitor_loop(args.monitor_interval, gpu_samples, stop_event)
        )
    connector = aiohttp.TCPConnector(limit=0)                                  
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        wall_start = time.perf_counter()
        for i, prompt in enumerate(prompts):
            payload = build_payload(args, prompt, use_stream=args.stream)
            tasks.append(
                asyncio.create_task(
                    worker(
                        sem, session, args.url, payload, i,
                        args.stream, args.request_timeout,
                    )
                )
            )
        results: List[RequestResult] = await asyncio.gather(*tasks)
        wall_end = time.perf_counter()
    if monitor_task is not None:
        stop_event.set()
        await monitor_task
    return results, wall_end - wall_start, gpu_samples

def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)

def print_report(results: List[RequestResult], wall_time: float, gpu_samples: List[GpuSample], args):
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    total_completion_tokens = sum(r.completion_tokens for r in successes)
    total_prompt_tokens = sum(r.prompt_tokens for r in successes)
    latencies = [r.latency for r in successes]
    per_req_tps = [r.tokens_per_sec for r in successes]
    ttfts = [r.ttft for r in successes if r.ttft is not None]
    W = 60
    print("\n" + "=" * W)
    print(" LLAMA.CPP PERFORMANCE REPORT".center(W))
    print("=" * W)
    print(f"Model:                 {args.model}")
    print(f"Quantization:          {args.llama_quant}")
    print(f"GPU Layers:            {args.llama_gpu_layers}")
    print(f"Context:               {args.llama_ctx_size}")
    print(f"Slots (--parallel):    {args.llama_slots}")
    print(f"Batch (--batch-size):  {args.llama_batch}")
    print(f"Micro Batch (--ubatch):{args.llama_ubatch}")
    print(f"Endpoint:              {args.url}")
    print("-" * W)
    print("REQUESTS")
    print(f"  Total Requests:      {len(results)}")
    print(f"  Concurrency:         {args.concurrency}")
    print(f"  Completed:           {len(successes)}")
    print(f"  Failed:              {len(failures)}")
    print(f"  Wall clock time:     {wall_time:.2f} s")
    print("-" * W)
    if successes:
        print("TOKENS")
        print(f"  Prompt Tokens:       {total_prompt_tokens}")
        print(f"  Generated Tokens:    {total_completion_tokens}")
        print(f"  Total Tokens:        {total_prompt_tokens + total_completion_tokens}")
        print("-" * W)
        print("THROUGHPUT")
        print(f"  Prompt tok/s:        {total_prompt_tokens / wall_time:.2f}")
        print(f"  Generation tok/s:    {total_completion_tokens / wall_time:.2f}")
        print(f"  Total tok/s:         {(total_prompt_tokens + total_completion_tokens) / wall_time:.2f}")
        print(f"  Requests/s:          {len(successes) / wall_time:.2f}")
        print("-" * W)
        print("LATENCY (seconds)")
        print(f"  Mean:                {statistics.mean(latencies):.3f}")
        print(f"  Median:              {statistics.median(latencies):.3f}")
        print(f"  P95:                 {percentile(latencies, 95):.3f}")
        print(f"  P99:                 {percentile(latencies, 99):.3f}")
        print(f"  Min / Max:           {min(latencies):.3f} / {max(latencies):.3f}")
        print("-" * W)
        print("PER REQUEST (tokens/sec)")
        print(f"  Mean tok/s:          {statistics.mean(per_req_tps):.2f}")
        print(f"  Median tok/s:        {statistics.median(per_req_tps):.2f}")
        fastest = max(successes, key=lambda r: r.tokens_per_sec)
        slowest = min(successes, key=lambda r: r.tokens_per_sec)
        print(f"  Fastest request:     #{fastest.idx} @ {fastest.tokens_per_sec:.2f} tok/s")
        print(f"  Slowest request:     #{slowest.idx} @ {slowest.tokens_per_sec:.2f} tok/s")
        if ttfts:
            print("-" * W)
            print("TIME TO FIRST TOKEN (seconds, streaming mode)")
            print(f"  Mean:                {statistics.mean(ttfts):.3f}")
            print(f"  P95:                 {percentile(ttfts, 95):.3f}")
    if gpu_samples:
        utils = [s.util_pct for s in gpu_samples]
        mems = [s.mem_used_mb for s in gpu_samples]
        powers = [s.power_w for s in gpu_samples]
        temps = [s.temp_c for s in gpu_samples]
        print("-" * W)
        print(f"SYSTEM (nvidia-smi, {len(gpu_samples)} samples @ {args.monitor_interval}s)")
        print(f"  GPU Util % (avg/max):  {statistics.mean(utils):.1f} / {max(utils):.1f}")
        print(f"  VRAM Used MB (avg/max):{statistics.mean(mems):.0f} / {max(mems):.0f}")
        print(f"  GPU Power W (avg/max): {statistics.mean(powers):.1f} / {max(powers):.1f}")
        print(f"  GPU Temp C (avg/max):  {statistics.mean(temps):.1f} / {max(temps):.1f}")
    elif args.monitor_gpu:
        print("-" * W)
        print("SYSTEM: nvidia-smi not found or unavailable -- no GPU stats collected")
    if failures:
        print("-" * W)
        print(f"FAILURES ({len(failures)}):")
        err_counts = {}
        for f in failures:
            err_counts[f.error] = err_counts.get(f.error, 0) + 1
        for err, count in sorted(err_counts.items(), key=lambda x: -x[1]):
            print(f"  {count}x  {err}")
    print("=" * W + "\n")
    if args.json_out:
        summary = {
            "engine": "llama.cpp",
            "url": args.url,
            "model": args.model,
            "quantization": args.llama_quant,
            "gpu_layers": args.llama_gpu_layers,
            "ctx_size": args.llama_ctx_size,
            "slots": args.llama_slots,
            "batch_size": args.llama_batch,
            "ubatch_size": args.llama_ubatch,
            "num_requests": len(results),
            "concurrency": args.concurrency,
            "wall_time_s": wall_time,
            "successful": len(successes),
            "failed": len(failures),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "prompt_tok_per_sec": total_prompt_tokens / wall_time if wall_time else 0,
            "generation_tok_per_sec": total_completion_tokens / wall_time if wall_time else 0,
            "total_tok_per_sec": (total_prompt_tokens + total_completion_tokens) / wall_time if wall_time else 0,
            "requests_per_sec": len(successes) / wall_time if wall_time else 0,
            "latency_mean_s": statistics.mean(latencies) if latencies else 0,
            "latency_median_s": statistics.median(latencies) if latencies else 0,
            "latency_p95_s": percentile(latencies, 95),
            "latency_p99_s": percentile(latencies, 99),
            "per_request_tok_per_sec_mean": statistics.mean(per_req_tps) if per_req_tps else 0,
            "gpu_util_avg": statistics.mean([s.util_pct for s in gpu_samples]) if gpu_samples else None,
            "gpu_util_max": max([s.util_pct for s in gpu_samples]) if gpu_samples else None,
            "vram_used_mb_avg": statistics.mean([s.mem_used_mb for s in gpu_samples]) if gpu_samples else None,
            "vram_used_mb_max": max([s.mem_used_mb for s in gpu_samples]) if gpu_samples else None,
            "gpu_power_w_avg": statistics.mean([s.power_w for s in gpu_samples]) if gpu_samples else None,
            "gpu_temp_c_avg": statistics.mean([s.temp_c for s in gpu_samples]) if gpu_samples else None,
            "per_request_results": [
                {
                    "idx": r.idx, "success": r.success, "latency_s": r.latency,
                    "completion_tokens": r.completion_tokens,
                    "prompt_tokens": r.prompt_tokens,
                    "tokens_per_sec": r.tokens_per_sec,
                    "ttft_s": r.ttft, "error": r.error,
                }
                for r in results
            ],
        }
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Detailed JSON report written to: {args.json_out}")

def build_config() -> SimpleNamespace:
    return SimpleNamespace(
        url=URL,
        model=MODEL,
        num_requests=NUM_REQUESTS,
        concurrency=CONCURRENCY,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        prompt=PROMPT,
        prompt_file=PROMPT_FILE,
        chat=CHAT,
        stream=STREAM,
        request_timeout=REQUEST_TIMEOUT,
        json_out=JSON_OUT,
        llama_quant=LLAMA_QUANT,
        llama_gpu_layers=LLAMA_GPU_LAYERS,
        llama_ctx_size=LLAMA_CTX_SIZE,
        llama_slots=LLAMA_SLOTS,
        llama_batch=LLAMA_BATCH,
        llama_ubatch=LLAMA_UBATCH,
        monitor_gpu=MONITOR_GPU,
        monitor_interval=MONITOR_INTERVAL,
    )

def main():
    args = build_config()
    if args.concurrency != args.llama_slots:
        print(
            f"[warning] CONCURRENCY ({args.concurrency}) != LLAMA_SLOTS "
            f"({args.llama_slots}). For accurate saturation testing these "
            f"should usually match your server's --parallel value.\n"
        )
    results, wall_time, gpu_samples = asyncio.run(run_benchmark(args))
    print_report(results, wall_time, gpu_samples, args)
if __name__ == "__main__":
    main()
