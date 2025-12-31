# Scripts Overview

This directory contains the core testing framework scripts.

## CLI Scripts

| Script | Description |
|--------|-------------|
| `run_test.py` | Single model test runner |
| `batch_test.py` | Batch test runner with cooldown |
| `analyze_results.py` | Results analysis and comparison |
| `score_results.py` | Interactive manual scoring |
| `llamacpp_batch.sh` | llama.cpp container management |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Single model test
python run_test.py --engine ollama --model qwen3:4b

# Batch test
python batch_test.py --engine ollama --config models.txt

# Analyze results
python analyze_results.py results/ --compare

# Score results
python score_results.py results/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all | GPUs to use |
| `OLLAMA_URL` | localhost:11434 | Ollama server URL |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error or incomplete |
| 130 | Interrupted (Ctrl+C) |

## Directory Structure

```
scripts/
├── run_test.py          # Single model testing
├── batch_test.py        # Batch testing with cooldown
├── analyze_results.py   # Results analysis
├── score_results.py     # Manual scoring
├── llamacpp_batch.sh    # llama.cpp wrapper
├── requirements.txt     # Python dependencies
├── core/                # Core modules
│   ├── questions.py     # Question loading
│   ├── results.py       # Results handling
│   ├── metrics.py       # GPU monitoring
│   └── runner.py        # Test execution
├── engines/             # Engine adapters
│   ├── ollama.py        # Ollama adapter
│   ├── llamacpp.py      # llama.cpp adapter
│   └── vllm.py          # vLLM adapter
└── data/                # Test data
    ├── homelab_inventory.json
    └── homelab_test_questions.json
```
