# Scripts Overview

This directory contains the core testing framework scripts.

## CLI Scripts

| Script | Description |
|--------|-------------|
| `preflight_check.py` | Pre-flight validation (connectivity, models, VRAM) |
| `run_test.py` | Single model test runner |
| `batch_test.py` | Batch test runner with cooldown and preflight |
| `analyze_results.py` | Results analysis and comparison |
| `score_results.py` | Interactive manual scoring |
| `llamacpp_batch.sh` | llama.cpp container management |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Pre-flight check
python preflight_check.py --engine ollama --config models.txt

# Single model test
python run_test.py --engine ollama --model qwen3:4b

# Batch test with preflight
python batch_test.py --engine ollama --config models.txt --preflight

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
| `LLAMACPP_URL` | localhost:8080 | llama.cpp server URL |
| `VLLM_URL` | localhost:8000 | vLLM server URL |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error or incomplete |
| 130 | Interrupted (Ctrl+C) |

## Directory Structure

```
scripts/
├── preflight_check.py   # Pre-flight validation CLI
├── run_test.py          # Single model testing
├── batch_test.py        # Batch testing with cooldown
├── analyze_results.py   # Results analysis
├── score_results.py     # Manual scoring
├── llamacpp_batch.sh    # llama.cpp wrapper
├── requirements.txt     # Python dependencies
├── core/                # Core modules
│   ├── __init__.py      # Module exports
│   ├── preflight.py     # Pre-flight check logic
│   ├── questions.py     # Question loading
│   ├── results.py       # Results handling
│   ├── metrics.py       # GPU monitoring
│   ├── runner.py        # Test execution
│   ├── config.py        # Configuration
│   ├── validation.py    # Input validation
│   └── logging_config.py # Logging setup
├── engines/             # Engine adapters
│   ├── __init__.py      # Module exports
│   ├── ollama.py        # Ollama adapter
│   ├── llamacpp.py      # llama.cpp adapter
│   └── vllm.py          # vLLM adapter
└── data/                # Test data (100 questions)
    ├── homelab_inventory.json
    └── homelab_test_questions.json
```
