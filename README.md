# Local LLM Testing Framework

A production-ready testing framework for evaluating and benchmarking small language models across different GPU configurations. Built for Proxmox VE environments with GPU passthrough to LXC containers, designed for testing models on consumer GPUs (starting with 6GB VRAM GTX 1660 Super) to find optimal models before fine-tuning.

## Test Environment

- **Host**: Proxmox VE with NVIDIA drivers
- **Containers**: LXC with GPU passthrough
- **Inference**: Docker containers (Ollama, llama.cpp, vLLM)
- **Storage**: Shared NVMe mount for models across containers

## Features

- **Multi-Engine Support**: Test with Ollama, llama.cpp, or vLLM
- **Multi-GPU Configurations**: Support for 1x, 2x, 4x GPU setups and mixed GPU configs
- **Resumable Testing**: Graceful shutdown with Ctrl+C, resume from last completed question
- **GPU Metrics**: Real-time VRAM, utilization, and temperature monitoring
- **Dynamic Cooldown**: Temperature-based cooldown between models
- **Scoring System**: Interactive manual scoring with 0-3 scale
- **Power Management**: Optional power limiting for thermal management

## Quick Start

### 1. Prerequisites

**Proxmox Host:**
- Proxmox VE 8.x with NVIDIA drivers installed
- GPU passthrough configured for LXC containers

**LXC Container:**
- Ubuntu 22.04/24.04 with GPU access
- Docker with NVIDIA Container Toolkit
- Python 3.9+

```bash
# In LXC container - verify GPU passthrough
nvidia-smi

# Install Python dependencies
pip install -r scripts/requirements.txt

# Start Ollama (Docker)
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

See [docs/SETUP.md](docs/SETUP.md) for detailed Proxmox and LXC configuration.

### 2. Run Single Model Test

```bash
cd llm-testing
python scripts/run_test.py --engine ollama --model qwen3:4b --gpu 1660super-1x
```

### 3. Run Batch Tests

```bash
python scripts/batch_test.py \
    --engine ollama \
    --config gpus/nvidia/1660super-1x/configs/ollama_models.txt \
    --results-dir gpus/nvidia/1660super-1x/results/ollama \
    --gpu 1660super-1x
```

### 4. Check Status

```bash
python scripts/batch_test.py \
    --config gpus/nvidia/1660super-1x/configs/ollama_models.txt \
    --results-dir gpus/nvidia/1660super-1x/results/ollama \
    --status
```

### 5. Analyze Results

```bash
python scripts/analyze_results.py gpus/nvidia/1660super-1x/results/ollama/ --compare
```

## Architecture

```
                                    ┌─────────────────┐
                                    │   User CLI      │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
            ┌───────▼───────┐       ┌────────▼────────┐      ┌────────▼────────┐
            │  run_test.py  │       │  batch_test.py  │      │ analyze_results │
            │  (single)     │       │  (batch)        │      │     .py         │
            └───────┬───────┘       └────────┬────────┘      └─────────────────┘
                    │                        │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     TestRunner        │
                    │   (core/runner.py)    │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐      ┌───────▼───────┐       ┌───────▼───────┐
│  OllamaAdapter│      │LlamaCppAdapter│       │  VLLMAdapter  │
│ (engines/)    │      │ (engines/)    │       │  (engines/)   │
└───────┬───────┘      └───────┬───────┘       └───────┬───────┘
        │                      │                       │
        ▼                      ▼                       ▼
   ┌─────────┐           ┌──────────┐            ┌─────────┐
   │ Ollama  │           │llama.cpp │            │  vLLM   │
   │ :11434  │           │  :8080   │            │  :8000  │
   └─────────┘           └──────────┘            └─────────┘
```

## Directory Structure

```
llm-testing/
├── README.md                              # This file
├── models/                                # Shared model storage
│   ├── ollama/                            # Ollama cache
│   ├── gguf/                              # GGUF files for llama.cpp
│   └── hf/                                # HuggingFace cache for vLLM
├── scripts/                               # Test framework code
│   ├── run_test.py                        # Single model test
│   ├── batch_test.py                      # Batch testing
│   ├── analyze_results.py                 # Results analysis
│   ├── score_results.py                   # Manual scoring
│   ├── llamacpp_batch.sh                  # llama.cpp wrapper
│   ├── core/                              # Core modules
│   ├── engines/                           # Engine adapters
│   └── data/                              # Test data
├── gpus/                                  # GPU-specific configs & results
│   └── nvidia/
│       ├── 1660super-1x/                  # Single GTX 1660 Super (6GB)
│       ├── 1660super-2x/                  # Dual GTX 1660 Super (12GB)
│       ├── 1660super-4x/                  # Quad GTX 1660 Super (24GB)
│       ├── 3060ti/                        # RTX 3060 Ti (8GB)
│       ├── 3070/                          # RTX 3070 (8GB)
│       ├── 3090ti/                        # RTX 3090 Ti (24GB)
│       └── 3090ti-3060ti/                 # Mixed config (32GB)
└── docs/                                  # Documentation
    ├── SETUP.md                           # Environment setup
    ├── USAGE.md                           # Usage examples
    └── TROUBLESHOOTING.md                 # Common issues
```

## Supported GPU Configurations

| GPU Config | Total VRAM | Model Size | Priority |
|------------|------------|------------|----------|
| GTX 1660 Super 1x | 6GB | 0.6B - 4B | Phase 1 |
| GTX 1660 Super 2x | 12GB | Up to 7B-13B Q4 | Phase 1 |
| GTX 1660 Super 4x | 24GB | Up to 32B Q4 | Phase 1 |
| RTX 3060 Ti | 8GB | Up to 7B Q4 | Phase 2 |
| RTX 3070 | 8GB | Up to 7B Q4 | Phase 2 |
| RTX 3090 Ti | 24GB | Up to 32B+ Q4 | Phase 3 |
| 3090 Ti + 3060 Ti | 32GB | Up to 70B Q4 | Phase 4 |

## Test Questions

The framework includes 42 sample test questions for a **homelab inventory assistant** use case, covering 8 categories:

1. **Simple Lookups (8)**: Direct retrieval (IP addresses, RAM, CPUs)
2. **Aggregation (6)**: Sum/count operations (total VRAM, system counts)
3. **Filtering (6)**: Conditional logic (systems with 10GbE, GPUs)
4. **Compatibility (6)**: Hardware relationships (RAM compatibility, upgrades)
5. **Complex Multi-step (4)**: Chained reasoning (best upgrade candidate)
6. **Natural Language (5)**: Informal query variations ("got anything beefy?")
7. **Inventory Management (3)**: Change/update scenarios
8. **Error Handling (4)**: Missing/invalid data responses

Custom question sets can be provided via `--questions` and `--inventory` flags for your own use cases.

## Scoring System

Manual scoring uses a 0-3 scale:
- **3**: Correct answer with relevant details
- **2**: Mostly correct, minor issues
- **1**: Partially correct or incomplete
- **0**: Wrong, hallucinated, or no answer

```bash
# Interactive scoring
python scripts/score_results.py gpus/nvidia/1660super-1x/results/ollama/

# View summary
python scripts/score_results.py gpus/nvidia/1660super-1x/results/ollama/ summary

# Export to CSV
python scripts/score_results.py gpus/nvidia/1660super-1x/results/ollama/ export
```

## CLI Reference

### run_test.py
```
python run_test.py --engine ENGINE --model MODEL [options]
  --engine, -e     Engine: ollama, llamacpp, vllm
  --model, -m      Model name
  --gpu, -g        GPU identifier (default: 1660super-1x)
  --results-dir    Results directory
  --max-tokens     Max tokens (default: 512)
  --temperature    Temperature (default: 0.1)
```

### batch_test.py
```
python batch_test.py --engine ENGINE --config FILE [options]
  --engine, -e     Engine: ollama, llamacpp, vllm
  --config, -c     Model list file
  --gpu, -g        GPU identifier
  --min-cooldown   Min cooldown seconds (default: 30)
  --target-temp    Target temp in C (default: 50)
  --status         Show progress only
  --power-limit    Power limits per GPU (e.g., "280,160")
```

### analyze_results.py
```
python analyze_results.py RESULTS_DIR [options]
  --detail         Detailed breakdown
  --compare        Compare multiple runs
  --export FILE    Export to CSV
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 16+ cores |
| RAM | 32GB | 64GB+ |
| GPU VRAM | 6GB | 8GB+ |
| Storage | 500GB SSD | 2TB+ NVMe |

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see docs/CONTRIBUTING.md for guidelines.
