# Usage Guide

This guide covers common workflows for the LLM Testing Framework.

## Quick Reference

```bash
# Pre-flight check before testing
python scripts/preflight_check.py --engine ollama --config models.txt

# Single model test
python scripts/run_test.py --engine ollama --model qwen3:4b

# Batch test with pre-flight
python scripts/batch_test.py --engine ollama --config models.txt --preflight

# Check status
python scripts/batch_test.py --config models.txt --status

# Score results
python scripts/score_results.py results/ollama/

# Analyze and compare
python scripts/analyze_results.py results/ollama/ --compare
```

## Pre-flight Checks

Before running tests, use the pre-flight system to validate model availability and engine connectivity.

### Standalone Pre-flight

```bash
# Check all models in a config file
python scripts/preflight_check.py --engine ollama --config gpus/nvidia/1660super-1x/configs/ollama_models.txt

# Check specific models
python scripts/preflight_check.py --engine ollama --models qwen3:4b llama3.2:3b smollm3:3b

# Full check including load test (loads model, sends test prompt)
python scripts/preflight_check.py --engine ollama --config models.txt --full

# Check all engines for a GPU config
python scripts/preflight_check.py --all-engines --gpu-config gpus/nvidia/1660super-1x/

# JSON output for scripting
python scripts/preflight_check.py --engine ollama --config models.txt --json
```

### Pre-flight CLI Options

| Flag | Description |
|------|-------------|
| `--engine`, `-e` | Engine to check: ollama, llamacpp, vllm |
| `--all-engines` | Check all engines |
| `--config`, `-c` | Path to model config file |
| `--models`, `-m` | Specific model names to check |
| `--gpu-config`, `-g` | GPU config directory (checks all engines) |
| `--full`, `-f` | Include load test (loads model, sends test prompt) |
| `--dry-run` | Show what would be checked without checking |
| `--json`, `-j` | Output results as JSON |
| `--quiet`, `-q` | Only show summary and errors |
| `--url`, `-u` | Override engine URL |
| `--timeout`, `-t` | Timeout for connectivity checks (seconds) |
| `--skip-disk-check` | Skip disk space verification |
| `--skip-vram-check` | Skip VRAM estimation check |
| `--models-dir` | Models directory for llama.cpp (default: /models) |

### Pre-flight Output Example

```
=== Pre-flight Check: Ollama ===

Engine Connectivity:
  ✓ Connected to http://localhost:11434 (Ollama v0.5.4)

Model Availability:
  ✓ qwen3:4b          [READY - CACHED]     VRAM: ~2.5GB
  ✓ smollm3:3b        [READY - CACHED]     VRAM: ~2.2GB
  ⚠ llama3.2:3b       [NEEDS DOWNLOAD]     Size: 1.9GB
  ✗ invalid-model:1b  [NOT FOUND]

Summary: 2 ready, 1 needs download, 1 failed
Overall: WARNING - Some models need download
```

### Integrated Pre-flight in Batch Testing

```bash
# Run pre-flight before starting tests
python scripts/batch_test.py --engine ollama --config models.txt --preflight

# Full pre-flight with load tests
python scripts/batch_test.py --engine ollama --config models.txt --preflight-full

# Only run pre-flight, don't start tests
python scripts/batch_test.py --engine ollama --config models.txt --preflight-only

# Skip VRAM estimation (faster)
python scripts/batch_test.py --engine ollama --config models.txt --preflight --skip-preflight-vram
```

### Pre-flight Check Levels

| Level | What's Checked | When to Use |
|-------|----------------|-------------|
| Basic | Engine connectivity, model existence | Quick validation |
| Standard | + VRAM estimation, disk space | Before batch runs |
| Full (`--full`) | + Load model, send test prompt | First-time setup |

## Single Model Testing

### Basic Test

```bash
python scripts/run_test.py \
    --engine ollama \
    --model qwen3:4b \
    --gpu 1660super-1x
```

### With Custom Options

```bash
python scripts/run_test.py \
    --engine ollama \
    --model qwen3:4b \
    --gpu 1660super-1x \
    --max-tokens 1024 \
    --temperature 0.2 \
    --warmup 2 \
    --results-dir ./my-results/
```

### With Extended Timeouts

For larger models that take longer to download/load:

```bash
python scripts/run_test.py \
    --engine ollama \
    --model mixtral:8x7b-instruct-q4_K_M \
    --download-timeout 3600 \
    --load-timeout 900
```

## Batch Testing

### Using Config File

```bash
python scripts/batch_test.py \
    --engine ollama \
    --config gpus/nvidia/1660super-1x/configs/ollama_models.txt \
    --results-dir gpus/nvidia/1660super-1x/results/ollama \
    --gpu 1660super-1x
```

### Using Model List

```bash
python scripts/batch_test.py \
    --engine ollama \
    --models qwen3:0.6b qwen3:1.7b qwen3:4b llama3.2:3b \
    --gpu 1660super-1x
```

### With Power Limiting

For high-power GPU configurations:

```bash
python scripts/batch_test.py \
    --engine ollama \
    --config gpus/nvidia/3090ti-3060ti/configs/ollama_models.txt \
    --gpu 3090ti-3060ti \
    --power-limit "280,160" \
    --power-threshold 800
```

### Dry Run

Preview what will be tested:

```bash
python scripts/batch_test.py \
    --engine ollama \
    --config models.txt \
    --dry-run
```

## Checking Status

### View Progress

```bash
python scripts/batch_test.py \
    --config gpus/nvidia/1660super-1x/configs/ollama_models.txt \
    --results-dir gpus/nvidia/1660super-1x/results/ollama \
    --status
```

Sample output:
```
Model                          Status       Progress   Latency      Tok/s      Updated
-------------------------------------------------------------------------------------
qwen3:0.6b                     Complete   100/100    245ms        73.5       2025-12-30 14:30
qwen3:4b                       Partial     53/100    523ms        48.2       2025-12-30 16:12
llama3.2:3b                    Pending      0/100    -            -          -
```

## Resuming Interrupted Tests

The framework automatically resumes from where it left off:

1. **Ctrl+C once**: Finishes current question, then exits
2. **Ctrl+C twice**: Force immediate exit
3. **Re-run same command**: Picks up from last completed question

```bash
# Original run (interrupted)
python scripts/batch_test.py --engine ollama --config models.txt

# Resume by running same command
python scripts/batch_test.py --engine ollama --config models.txt
```

## llama.cpp Testing

Since llama.cpp loads models at startup, use the batch wrapper:

```bash
./scripts/llamacpp_batch.sh \
    gpus/nvidia/1660super-1x/configs/llamacpp_models.txt \
    gpus/nvidia/1660super-1x/results/llamacpp \
    --gpu 1660super-1x
```

### Multi-GPU llama.cpp

```bash
./scripts/llamacpp_batch.sh \
    gpus/nvidia/1660super-2x/configs/llamacpp_models.txt \
    gpus/nvidia/1660super-2x/results/llamacpp \
    --tensor-split "0.5,0.5" \
    --gpu 1660super-2x
```

## Analyzing Results

### Single Model Analysis

```bash
python scripts/analyze_results.py \
    gpus/nvidia/1660super-1x/results/ollama/qwen3-4b/ \
    --detail
```

### Compare All Models

```bash
python scripts/analyze_results.py \
    gpus/nvidia/1660super-1x/results/ollama/ \
    --compare
```

### Export to CSV

```bash
python scripts/analyze_results.py \
    gpus/nvidia/1660super-1x/results/ollama/ \
    --compare \
    --export comparison.csv
```

### JSON Output

```bash
python scripts/analyze_results.py \
    gpus/nvidia/1660super-1x/results/ollama/qwen3-4b/ \
    --json
```

## Scoring Results

### Interactive Scoring

```bash
python scripts/score_results.py \
    gpus/nvidia/1660super-1x/results/ollama/
```

The scorer displays:
```
Question 44/100 [compatibility]
Q: What RAM is compatible with the ASUS PRIME B550-PLUS?
Expected: DDR4 3200MHz 32GB modules (2 available in spare parts)

Model Response:
The ASUS PRIME B550-PLUS is compatible with DDR4 RAM...

Score (0-3, s=skip, q=quit):
```

### Score Specific Model

```bash
python scripts/score_results.py \
    gpus/nvidia/1660super-1x/results/ollama/ \
    --model qwen3-4b
```

### Score Specific Category

```bash
python scripts/score_results.py \
    gpus/nvidia/1660super-1x/results/ollama/ \
    --category compatibility
```

### Re-score Already Scored

```bash
python scripts/score_results.py \
    gpus/nvidia/1660super-1x/results/ollama/ \
    --rescore
```

### View Scoring Summary

```bash
python scripts/score_results.py \
    gpus/nvidia/1660super-1x/results/ollama/ \
    summary
```

### Export Scores

```bash
python scripts/score_results.py \
    gpus/nvidia/1660super-1x/results/ollama/ \
    export --output scores.csv
```

## Comparing GPUs

Run same models on different GPUs, then compare:

```bash
# Test on 1660 Super
python scripts/batch_test.py \
    --engine ollama \
    --config gpus/nvidia/1660super-1x/configs/ollama_models.txt \
    --results-dir gpus/nvidia/1660super-1x/results/ollama \
    --gpu 1660super-1x

# Test on 3060 Ti
python scripts/batch_test.py \
    --engine ollama \
    --config gpus/nvidia/3060ti/configs/ollama_models.txt \
    --results-dir gpus/nvidia/3060ti/results/ollama \
    --gpu 3060ti

# Compare all NVIDIA GPU results
python scripts/analyze_results.py gpus/nvidia/ --compare
```

## Custom Test Data

### Using Custom Questions

```bash
python scripts/run_test.py \
    --engine ollama \
    --model qwen3:4b \
    --questions /path/to/my_questions.json \
    --inventory /path/to/my_inventory.json
```

### Question File Format

```json
{
  "questions": [
    {
      "q_id": 1,
      "category": "simple_lookup",
      "question": "What is the IP of server-01?",
      "expected": "192.168.1.10",
      "difficulty": "easy"
    }
  ]
}
```

## Environment Variables

```bash
# Override default URLs
export OLLAMA_URL=http://192.168.1.100:11434
export LLAMACPP_URL=http://192.168.1.100:8080
export VLLM_URL=http://192.168.1.100:8000

# Specify GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

## Common Workflows

### New GPU Configuration Test

1. Create config file:
   ```bash
   cp gpus/nvidia/1660super-1x/configs/ollama_models.txt \
      gpus/nvidia/my-gpu/configs/ollama_models.txt
   # Edit to include models that fit in your GPU's VRAM
   ```

2. Create results directories:
   ```bash
   mkdir -p gpus/nvidia/my-gpu/results/{ollama,llamacpp,vllm}
   ```

3. Run tests:
   ```bash
   python scripts/batch_test.py \
       --engine ollama \
       --config gpus/nvidia/my-gpu/configs/ollama_models.txt \
       --results-dir gpus/nvidia/my-gpu/results/ollama \
       --gpu my-gpu
   ```

### Full Benchmark Cycle

```bash
# 1. Pre-flight check
python scripts/preflight_check.py --engine ollama --config models.txt

# 2. Run all models with pre-flight
python scripts/batch_test.py --engine ollama --config models.txt --preflight

# 3. Score results interactively
python scripts/score_results.py results/ollama/

# 4. Analyze and export
python scripts/analyze_results.py results/ollama/ --compare --export results.csv

# 5. View final summary
python scripts/score_results.py results/ollama/ summary
```

## Output Files

### Results Directory Structure

```
gpus/nvidia/1660super-1x/results/ollama/
├── qwen3-4b/
│   ├── run_metadata.json      # Test run configuration
│   ├── questions.jsonl        # Per-question results (append-only)
│   └── scores.json            # Manual scoring data
├── smollm3-3b/
│   ├── run_metadata.json
│   ├── questions.jsonl
│   └── scores.json
└── failures.log               # Model load failures
```

### File Formats

**run_metadata.json** - Test configuration:
```json
{
  "model": "qwen3:4b",
  "engine": "ollama",
  "gpu": "1660super-1x",
  "started_at": "2025-12-30T14:30:00",
  "questions_file": "homelab_test_questions.json",
  "total_questions": 100
}
```

**questions.jsonl** - Per-question results (one JSON object per line):
```json
{"q_id": 1, "question": "What is...", "response": "The answer...", "latency_ms": 523, "tokens": 48, "timestamp": "..."}
```

**scores.json** - Manual scoring data:
```json
{
  "scores": {"1": 3, "2": 2, "3": 3, ...},
  "total": 285,
  "max_possible": 300,
  "scored_count": 100
}
```

## vLLM Memory Tuning for Low VRAM GPUs

### Understanding vLLM Memory Usage

vLLM memory = Model Weights + CUDA Overhead + Activations + KV Cache

Unlike Ollama/llama.cpp which allocate memory on-demand, vLLM pre-allocates the KV cache at startup. This means:

- A 1B model on a 24GB GPU will still use ~22GB (model + pre-allocated KV cache)
- Reducing `gpu_memory_utilization` limits KV cache but reduces max concurrent requests
- `max_model_len` directly impacts KV cache size

### Key Parameters

| Parameter | Effect | Recommendation for 6GB |
|-----------|--------|------------------------|
| `gpu_memory_utilization` | % of VRAM to use | 0.95 (max out) |
| `max_model_len` | Max context length | 1024-2048 |
| `enforce_eager` | Disable CUDA graphs | True (saves ~500MB) |
| `max_num_seqs` | Max concurrent sequences | 4-8 |
| `max_num_batched_tokens` | Tokens per batch | 2048-4096 |

### Quantization Options

vLLM supports AWQ and GPTQ quantized models from HuggingFace:

```bash
# AWQ model
vllm serve TheBloke/Qwen-4B-AWQ --quantization awq

# GPTQ model
vllm serve TheBloke/Qwen-4B-GPTQ --quantization gptq
```

### Starting vLLM for 6GB GPUs

```bash
# Docker with optimized settings
docker run -d --gpus all \
    -v /mnt/llm-testing/models/hf:/root/.cache/huggingface \
    -p 8000:8000 \
    --name vllm \
    vllm/vllm-openai \
    --model Qwen/Qwen2.5-3B-Instruct-AWQ \
    --quantization awq \
    --gpu-memory-utilization 0.95 \
    --max-model-len 2048 \
    --enforce-eager \
    --max-num-seqs 4
```

### Troubleshooting OOM Errors

If you get `CUDA out of memory`:

1. **Reduce `max_model_len`** - Halving it roughly doubles available KV cache space
2. **Add `--enforce-eager`** - Disables CUDA graphs, saves ~500MB
3. **Reduce `max_num_seqs`** - Fewer concurrent sequences
4. **Try a smaller/more quantized model** - AWQ models use ~4x less memory
5. **Check for other GPU processes** - Run `nvidia-smi` to verify

### When to Use vLLM vs Ollama

| Use Case | Recommended Engine |
|----------|-------------------|
| Single user, interactive | Ollama |
| Batch processing many queries | vLLM |
| API server, multiple users | vLLM |
| Maximum VRAM efficiency | llama.cpp |
| Quick model testing | Ollama |
| Production deployment | llama.cpp or vLLM |

### vLLM Memory Calculator

Rough formula for 4-bit quantized models:

```
Required VRAM (GB) ≈ (Parameters in B × 0.5) + 2 + (max_model_len × 0.001)

Example: Qwen3-4B AWQ with 2048 context
= (4 × 0.5) + 2 + (2048 × 0.001)
= 2 + 2 + 2
= ~6GB (barely fits)
```

For FP16 models, multiply parameter estimate by 2.
