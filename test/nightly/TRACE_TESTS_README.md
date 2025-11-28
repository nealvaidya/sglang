# NeurIPS MoE Performance Trace Tests

Individual trace tests for benchmarking large MoE models on H200 with native precision.

## Test Files

All tests output to the same directory: `performance_profiles_neurips_traces/`

- `trace_qwen3_235b.py` - Qwen3-235B (TP4, TP8) × (auto, triton)
- `trace_qwen3_coder_480b.py` - Qwen3-Coder-480B (TP4, TP8) × (auto, triton)
- `trace_deepseek_v31.py` - DeepSeek V3.1 (TP4, TP8) × (auto, triton)
- `trace_minimax_m2.py` - MiniMax-M2 (TP1, TP2, TP8) × (auto, triton)
- `trace_kimi_k2.py` - Kimi-K2-Thinking (TP1, TP2, TP8) × (auto, triton)
- `trace_glm_46.py` - GLM-4.6 (TP8) × (auto, triton)

## Results Tracking

**All tests share a centralized results tracker** that maintains:

### `success_results.json`
Stores successful benchmark results with key format `model::config`:
```json
{
  "Qwen3-235B::TP8_EP2_auto": {
    "model": "Qwen3-235B",
    "config": "TP8_EP2_auto",
    "batch_size": 1,
    "input_len": 4096,
    "output_len": 512,
    "latency_s": 7.59,
    "input_throughput_tok_per_s": 15933.87,
    "output_throughput_tok_per_s": 69.81,
    "itl_ms": 14.32,
    "last_updated": "2025-11-27T23:45:12"
  }
}
```

### `failure_results.json`
Stores failed configurations with concise error messages:
```json
{
  "DeepSeek-V3.1::TP4_auto": {
    "model": "DeepSeek-V3.1",
    "config": "TP4_auto",
    "error_message": "CUDA out of memory...",
    "error_type": "OOM",
    "last_updated": "2025-11-27T23:50:00"
  }
}
```

**Key Features:**
- ✅ Results persist across runs - re-running updates existing entries
- ✅ Successful runs remove entries from failures
- ✅ Failed runs don't overwrite successful results
- ✅ Automatic error type classification (OOM, Timeout, ServerError, etc.)

## Running Tests

Run each test individually on your H200 cluster:

```bash
# Example: Run Qwen3-235B trace
cd /sgl-workspace/root/sglang/test
python3 nightly/trace_qwen3_235b.py
```

Each test will:
1. Launch the model with specified TP/EP/MoE backend configs
2. Run batch-1 benchmark (4096 input tokens, 512 output tokens)
3. Generate performance traces
4. Print a markdown table with metrics

## Output Format

Each test prints a table like:

```
Model (variant)
batch size | input len | latency (s) | input throughput (tok/s) | output throughput (tok/s) | ITL (ms) | profile (extend) | profile (decode)
1 | 4096 | 7.59 | 15933.87 | 69.81 | 14.32 | trace | trace
```

## Notes

- **TP4 configs** on large models (DeepSeek, Qwen-480B) will likely OOM - expected behavior
- **TP8 with EP=2** should work for all models
- Model weights are cached after first download, so re-runs are fast
- All traces save to `performance_profiles_neurips_traces/` for easy collection

## Customizing Configs

Edit the `CONFIGS` list in each file to test different TP/EP/backend combinations:

```python
CONFIGS = [
    (4, 1, ["auto"]),  # TP4, EP1, auto backend
    (8, 2, ["auto", "triton_kernel"]),  # TP8, EP2, test multiple backends
]
```

Available MoE backends for native precision:
- `auto` (default - automatically selects best backend)
- `triton` (Triton fused MoE)
- `triton_kernel` (alternative Triton implementation)
- `deep_gemm`

## Viewing Results

After running tests, check the results:

```bash
# View success results
cat performance_profiles_neurips_traces/success_results.json | jq

# View failures
cat performance_profiles_neurips_traces/failure_results.json | jq

# Quick summary
python3 -c "
import json
success = json.load(open('performance_profiles_neurips_traces/success_results.json'))
failure = json.load(open('performance_profiles_neurips_traces/failure_results.json'))
print(f'✓ Successful: {len(success)}')
print(f'⚠ Failed: {len(failure)}')
"
```

Each test also prints a markdown table at the end showing all successful results grouped by model.
