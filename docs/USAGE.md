# Usage Guide

This guide covers common workflows for the LLM Testing Framework.

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
qwen3:0.6b                     Complete   42/42      245ms        73.5       2025-12-30 14:30
qwen3:4b                       Partial    23/42      523ms        48.2       2025-12-30 16:12
llama3.2:3b                    Pending    0/42       -            -          -
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
Question 15/42 [compatibility]
Q: What RAM is compatible with the ASUS PRIME B550-PLUS?
Expected: DDR4 3200MHz (32GB modules available in spare_parts)

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
# 1. Run all models
python scripts/batch_test.py --engine ollama --config models.txt

# 2. Score results
python scripts/score_results.py results/ollama/

# 3. Analyze and export
python scripts/analyze_results.py results/ollama/ --compare --export results.csv

# 4. View summary
python scripts/score_results.py results/ollama/ summary
```
