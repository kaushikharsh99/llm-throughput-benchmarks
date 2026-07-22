import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional
import aiohttp
# CONFIG - edit these values directly, then just run: python vllm_benchmark.py
URL = "http://localhost:8000/v1/chat/completions"                     
MODEL           = "cyankiwi/Qwen3.5-2B-AWQ-4bit"                                                          
NUM_REQUESTS    = 565                                       
CONCURRENCY     = 113                                                            
MAX_TOKENS      = 7000                             
TEMPERATURE     = 0.9
PROMPT          = None
PROMPT_FILE     = None                                                                                                    
CHAT            = True                                                                       
STREAM          = False                                                      
REQUEST_TIMEOUT = 300.0                                   
JSON_OUT        = None                                                                            
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
    full_text_chunks = []
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
                        full_text_chunks.append(choice["text"])
                        completion_tokens += 1                                         
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        full_text_chunks.append(delta["content"])
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
async def run_benchmark(args):
    prompts = load_prompts(args)
    sem = asyncio.Semaphore(args.concurrency)
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
    return results, wall_end - wall_start

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

def print_report(results: List[RequestResult], wall_time: float, args):
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    total_completion_tokens = sum(r.completion_tokens for r in successes)
    total_prompt_tokens = sum(r.prompt_tokens for r in successes)
    latencies = [r.latency for r in successes]
    per_req_tps = [r.tokens_per_sec for r in successes]
    ttfts = [r.ttft for r in successes if r.ttft is not None]
    print("\n" + "=" * 60)
    print(" vLLM BENCHMARK REPORT")
    print("=" * 60)
    print(f"Endpoint:              {args.url}")
    print(f"Model:                 {args.model}")
    print(f"Requests sent:         {len(results)}")
    print(f"Concurrency:           {args.concurrency}")
    print(f"Successful:            {len(successes)}")
    print(f"Failed:                {len(failures)}")
    print(f"Wall clock time:       {wall_time:.2f} s")
    print("-" * 60)
    if successes:
        print(f"Total prompt tokens:   {total_prompt_tokens}")
        print(f"Total completion tok:  {total_completion_tokens}")
        print(f"Total tokens:          {total_prompt_tokens + total_completion_tokens}")
        print("-" * 60)
        print("THROUGHPUT (aggregate, across all concurrent requests)")
        print(f"  Completion tok/sec:  {total_completion_tokens / wall_time:.2f}")
        print(f"  Total tok/sec:       {(total_prompt_tokens + total_completion_tokens) / wall_time:.2f}")
        print(f"  Requests/sec:        {len(successes) / wall_time:.2f}")
        print("-" * 60)
        print("PER-REQUEST LATENCY (seconds)")
        print(f"  Mean:                {statistics.mean(latencies):.3f}")
        print(f"  Median:              {statistics.median(latencies):.3f}")
        print(f"  P90:                 {percentile(latencies, 90):.3f}")
        print(f"  P99:                 {percentile(latencies, 99):.3f}")
        print(f"  Min / Max:           {min(latencies):.3f} / {max(latencies):.3f}")
        print("-" * 60)
        print("PER-REQUEST THROUGHPUT (tokens/sec, individual request)")
        print(f"  Mean:                {statistics.mean(per_req_tps):.2f}")
        print(f"  Median:              {statistics.median(per_req_tps):.2f}")
        print(f"  Min / Max:           {min(per_req_tps):.2f} / {max(per_req_tps):.2f}")
        if ttfts:
            print("-" * 60)
            print("TIME TO FIRST TOKEN (seconds, streaming mode)")
            print(f"  Mean:                {statistics.mean(ttfts):.3f}")
            print(f"  P90:                 {percentile(ttfts, 90):.3f}")
    if failures:
        print("-" * 60)
        print(f"FAILURES ({len(failures)}):")
        err_counts = {}
        for f in failures:
            err_counts[f.error] = err_counts.get(f.error, 0) + 1
        for err, count in sorted(err_counts.items(), key=lambda x: -x[1]):
            print(f"  {count}x  {err}")
    print("=" * 60 + "\n")
    if args.json_out:
        summary = {
            "url": args.url,
            "model": args.model,
            "num_requests": len(results),
            "concurrency": args.concurrency,
            "wall_time_s": wall_time,
            "successful": len(successes),
            "failed": len(failures),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "aggregate_completion_tok_per_sec": total_completion_tokens / wall_time if wall_time else 0,
            "aggregate_total_tok_per_sec": (total_prompt_tokens + total_completion_tokens) / wall_time if wall_time else 0,
            "requests_per_sec": len(successes) / wall_time if wall_time else 0,
            "latency_mean_s": statistics.mean(latencies) if latencies else 0,
            "latency_median_s": statistics.median(latencies) if latencies else 0,
            "latency_p90_s": percentile(latencies, 90),
            "latency_p99_s": percentile(latencies, 99),
            "per_request_tok_per_sec_mean": statistics.mean(per_req_tps) if per_req_tps else 0,
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
    )

def main():
    args = build_config()
    results, wall_time = asyncio.run(run_benchmark(args))
    print_report(results, wall_time, args)
if __name__ == "__main__":
    main()
