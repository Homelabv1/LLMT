# Local LLM Testing Framework

A production-ready testing framework for evaluating and benchmarking small language models across different GPU configurations. Built for Proxmox VE environments with GPU passthrough to LXC containers, designed for testing models on consumer GPUs (starting with 6GB VRAM GTX 1660 Super) to find optimal models before fine-tuning.

## Table of Contents

- [Test Environment](#test-environment)
- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Supported GPU Configurations](#supported-gpu-configurations)
- [Supported Models](#supported-models)
- [Test Questions](#test-questions)
- [Scoring System](#scoring-system)
- [CLI Reference](#cli-reference)
- [Engine-Specific Notes](#engine-specific-notes)
- [Hardware Requirements](#hardware-requirements)
- [Documentation](#documentation)
- [License](#license)

## Test Environment

- **Host**: Proxmox VE with NVIDIA drivers
- **Containers**: LXC with GPU passthrough
- **Inference**: Docker containers (Ollama, llama.cpp, vLLM)
- **Storage**: Shared NVMe mount for models across containers

## Features

- **Multi-Engine Support**: Test with Ollama, llama.cpp, or vLLM
- **Multi-GPU Configurations**: Support for 1x, 2x, 4x GPU setups and mixed GPU configs
- **Pre-flight Checks**: Validate model availability and engine connectivity before testing
- **Resumable Testing**: Graceful shutdown with Ctrl+C, resume from last completed question
- **GPU Metrics**: Real-time VRAM, utilization, and temperature monitoring
- **Dynamic Cooldown**: Temperature-based cooldown between models
- **Scoring System**: Interactive manual scoring with 0-3 scale (max 300 points for 100 questions)
- **Power Management**: Optional power limiting for thermal management

## Quick Start

### 1. Prerequisites

**Proxmox Host:**
- Proxmox VE 8.x with NVIDIA drivers installed
- GPU passthrough configured for LXC containers

**LXC Container:**
- Ubuntu 22.04/24.04 with GPU access
- Git (for cloning the repository)
- Docker with NVIDIA Container Toolkit
- Python 3.9+

```bash
# In LXC container - install prerequisites
apt update && apt install -y git python3 python3-pip

# Verify GPU passthrough
nvidia-smi

# Clone the repository
git clone <repository-url> llm-testing
cd llm-testing

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

## Supported Models

Models are organized by priority tier for 6GB VRAM testing:

### Tier 1 - Priority (December 2025)

| Model | Parameters | VRAM (Q4) | Key Features |
|-------|------------|-----------|--------------|
| SmolLM3-3B | 3B | ~2.2GB | Dual-mode reasoning, 128K context |
| Granite 4.0-H-Tiny | 7B (1B active) | ~1.5GB | Mamba-2 hybrid, 70% less RAM |
| Granite 4.0-H-Micro | 3B | ~2GB | Dense hybrid, tool calling |
| Qwen3-4B | 4B | ~2.5GB | 256K context, mature ecosystem |
| Gemma 3 4B | 4B | ~2GB | QAT optimized, multimodal |

### Tier 2 - Strong Candidates

| Model | Parameters | VRAM (Q4) | Key Features |
|-------|------------|-----------|--------------|
| Ministral 3B | 3.4B | ~2.5GB | Edge-optimized, function calling |
| Llama 3.2 3B | 3B | ~2GB | Meta baseline |
| Phi-4 Mini | 3.8B | ~2.2GB | Microsoft efficient model |

### Tier 3 - Baselines

| Model | Parameters | VRAM (Q4) | Purpose |
|-------|------------|-----------|---------|
| Qwen3-1.7B | 1.7B | ~1.2GB | Mid-size baseline |
| Qwen3-0.6B | 0.6B | ~0.5GB | Speed baseline |
| Granite Nano 1B | 1B | ~0.8GB | Edge optimized |
| TinyLlama 1.1B | 1.1B | ~0.8GB | Legacy baseline |

See [docs/MODEL_RESEARCH.md](docs/MODEL_RESEARCH.md) for detailed model analysis.

## Test Questions

The framework includes 100 sample test questions for a **homelab inventory assistant** use case, covering 8 categories:

1. **Simple Lookups (15)**: Direct retrieval (IP addresses, RAM, CPUs)
2. **Aggregation (14)**: Sum/count operations (total VRAM, system counts)
3. **Filtering (14)**: Conditional logic (systems with 10GbE, GPUs)
4. **Compatibility (12)**: Hardware relationships (RAM compatibility, upgrades)
5. **Complex Multi-step (12)**: Chained reasoning (best upgrade candidate)
6. **Natural Language (14)**: Informal query variations ("got anything beefy?")
7. **Inventory Management (10)**: Change/update scenarios
8. **Error Handling (9)**: Missing/invalid data responses

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

### preflight_check.py
```
python preflight_check.py --engine ENGINE [options]
  --engine, -e     Engine: ollama, llamacpp, vllm
  --all-engines    Check all engines
  --config, -c     Path to model config file
  --models, -m     Specific model names to check
  --gpu-config     GPU config directory
  --full           Include load test (loads model, sends test prompt)
  --json           Output results as JSON
  --skip-vram-check   Skip VRAM estimation
```

### run_test.py
```
python run_test.py --engine ENGINE --model MODEL [options]
  --engine, -e     Engine: ollama, llamacpp, vllm
  --model, -m      Model name
  --gpu, -g        GPU identifier (default: 1660super-1x)
  --results-dir    Results directory
  --max-tokens     Max tokens (default: 512)
  --temperature    Temperature (default: 0.1)
  --download-timeout  Download timeout (default: 1800s)
  --load-timeout   Load timeout (default: 600s)
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
  --preflight      Run pre-flight checks before testing
  --preflight-full Run full pre-flight including load tests
  --preflight-only Only run pre-flight, skip tests
```

### analyze_results.py
```
python analyze_results.py RESULTS_DIR [options]
  --detail         Detailed breakdown
  --compare        Compare multiple runs
  --export FILE    Export to CSV
  --by-category    Break down by question category
  --json           Output as JSON
```

## Engine-Specific Notes

### Ollama
Best for quick testing and single-user scenarios. Automatically manages model loading/unloading. Most VRAM-efficient for interactive use.

### llama.cpp
Maximum VRAM efficiency with GGUF quantized models. Requires manual model file management. Best for production deployments.

### vLLM Considerations

vLLM has significantly higher VRAM requirements than Ollama or llama.cpp due to:

1. **KV Cache Pre-allocation**: vLLM reserves ~90% of GPU memory by default
2. **CUDA/PyTorch Overhead**: ~0.5-1GB for buffers and memory management
3. **Activation Memory**: ~0.2-0.5GB depending on batch size

#### Realistic 6GB VRAM Budget for vLLM

| Component | Size |
|-----------|------|
| CUDA/driver overhead | 0.5-1GB |
| Activation memory | 0.2-0.5GB |
| Minimum KV cache | 0.5-1GB |
| **Available for model** | **~3.5-4.5GB** |

#### What Fits on 6GB with vLLM

| Model | Quantization | Fits? | Notes |
|-------|--------------|-------|-------|
| 1.5B | FP16 | ⚠️ | Tight, very limited context |
| 3B | FP16 | ❌ | No room for KV cache |
| 3B | AWQ/GPTQ 4-bit | ✅ | ~1.5GB weights, comfortable |
| 4B | AWQ/GPTQ 4-bit | ✅ | ~2GB weights, good fit |
| 7B | AWQ/GPTQ 4-bit | ❌ | ~3.5GB weights, no room for KV |

#### Recommended vLLM Parameters for 6GB

```bash
vllm serve <model> \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2048 \
  --enforce-eager \
  --max-num-seqs 4
```

**Note**: For single-user homelab assistant use cases, Ollama or llama.cpp may be more VRAM-efficient. vLLM excels at high-throughput batched inference.

See [docs/USAGE.md](docs/USAGE.md) for detailed vLLM tuning guide.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 16+ cores |
| RAM | 32GB | 64GB+ |
| GPU VRAM | 6GB | 8GB+ |
| Storage | 500GB SSD | 2TB+ NVMe |

## Documentation

| Document | Description |
|----------|-------------|
| [SETUP.md](docs/SETUP.md) | Environment setup (Proxmox, LXC, Docker, engines) |
| [USAGE.md](docs/USAGE.md) | Usage guide, CLI reference, vLLM tuning |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [MODEL_RESEARCH.md](docs/MODEL_RESEARCH.md) | December 2025 model research notes |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Contribution guidelines |
| [CONFIG_AUDIT_REPORT.md](docs/CONFIG_AUDIT_REPORT.md) | Model config inheritance validation |

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.
