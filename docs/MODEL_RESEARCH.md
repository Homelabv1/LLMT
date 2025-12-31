# Model Research Notes (December 2025)

This document contains research findings on models suitable for testing on consumer GPUs (6GB-24GB VRAM).

## Summary: Best Models for 6GB VRAM

| Model | Size | VRAM (Q4) | Strength | Priority |
|-------|------|-----------|----------|----------|
| SmolLM3-3B | 3B | ~2.2GB | Dual-mode reasoning, 128K context | 1 |
| Granite 4.0-H-Tiny | 7B (1B active) | ~1.5GB | Hybrid arch, 70% less RAM | 1 |
| Qwen3-4B (2507) | 4B | ~2.5GB | 256K context, mature ecosystem | 1 |
| Granite 4.0-H-Micro | 3B | ~2GB | Dense hybrid, tool calling | 1 |
| Gemma 3 4B QAT | 4B | ~2GB | Multimodal, QAT optimized | 2 |
| Ministral 3B | 3.4B | ~2.5GB | Edge-optimized, function calling | 2 |

## Detailed Model Analysis

### SmolLM3-3B (HuggingFace, July 2025)

**Overview:**
- **Developer**: HuggingFace
- **Release**: July 2025
- **Architecture**: Decoder-only transformer with GQA and NoPE
- **Training**: 11.2T tokens, fully open training data and configs

**Key Features:**
- **Dual-mode reasoning**: `/think` for complex queries, `/no_think` for fast responses
- **128K context**: 64K native, YaRN extension to 128K
- **Fully open**: Training data, configs, and weights all publicly available

**Architecture Details:**
- Grouped Query Attention (GQA) for efficiency
- NoPE (no positional encoding) on every 4th layer
- Optimized for inference on consumer hardware

**Benchmark Performance:**
- Outperforms Llama 3.2 3B and Qwen2.5 3B on most benchmarks
- Strong on IFEval, GSM8K, and HellaSwag
- Multilingual support (18 languages)

**VRAM Requirements:**
| Quantization | VRAM | Context |
|--------------|------|---------|
| Q4_K_M | ~2.2GB | 4K-8K |
| Q5_K_M | ~2.6GB | 4K |
| FP16 | ~6GB | 2K |

**Best For:**
- Complex reasoning when `/think` enabled
- Fast responses with `/no_think`
- Homelab assistant with reasoning explanations

**Ollama Tag:** `smollm3:3b`
**GGUF Source:** `ggml-org/SmolLM3-3B-GGUF`

---

### IBM Granite 4.0 (October-November 2025)

**Overview:**
- **Developer**: IBM
- **Release**: October-November 2025
- **Architecture**: Hybrid Mamba-2/Transformer (9:1 ratio), MoE variants
- **License**: Apache 2.0

**Key Innovation:**
The hybrid Mamba-2/Transformer architecture achieves **70% less RAM** than conventional transformers due to Mamba's linear scaling with sequence length.

**Model Variants:**

| Model | Total Params | Active Params | VRAM (Q4) | Notes |
|-------|--------------|---------------|-----------|-------|
| H-Small | 32B | 9B | ~4.5GB | Too large for 6GB FP16 |
| H-Tiny | 7B | 1B | ~0.8GB | Excellent for 6GB! |
| H-Micro | 3B | 3B (dense) | ~1.8GB | Dense hybrid |
| Micro | 3B | 3B | ~1.8GB | Transformer-only |
| Nano 1.5B | 1.5B | 1.5B | ~1GB | Edge optimized |
| Nano 350M | 350M | 350M | ~0.3GB | Smallest, baseline |

**H-Tiny Architecture:**
- MoE with 40 experts, 8 active per token
- 9:1 Mamba-2 to Transformer layer ratio
- Only 1B parameters active during inference

**Features:**
- Built-in tool calling and function execution
- ISO 42001 AI Management System certified
- Strong on IFEval, GPQA, GSM8K

**Best For:**
- Tool calling and structured outputs
- Enterprise deployments with compliance requirements
- Very low VRAM environments (H-Tiny uses ~1GB at Q4!)

**Ollama Tags:** `granite4-tiny:latest`, `granite4-micro:latest`
**GGUF Source:** `ibm-granite/granite-4.0-*-preview-gguf`

---

### Qwen3-4B (Updated July 2025 - 2507 version)

**Overview:**
- **Developer**: Alibaba Qwen Team
- **Release**: Original March 2025, updated July 2025
- **Architecture**: Dense transformer with RoPE

**July 2025 Updates:**
- Improved instruction following
- Better general capabilities
- Claims to rival Qwen2.5-72B on some benchmarks

**Key Features:**
- **256K context**: Native long context support
- **Dual-mode reasoning**: `/think` and `/no_think` modes
- **Mature ecosystem**: Wide tool support, fine-tuning resources

**VRAM Requirements:**
| Quantization | VRAM | Context |
|--------------|------|---------|
| Q4_K_M | ~2.5GB | 8K |
| Q5_K_M | ~3GB | 4K |
| AWQ | ~2.5GB | 4K (vLLM) |
| FP16 | ~8GB | 2K |

**Best For:**
- General purpose assistant
- Long context tasks
- Production deployment (mature ecosystem)

**Ollama Tag:** `qwen3:4b`
**GGUF Source:** `Qwen/Qwen3-4B-GGUF`, `bartowski/Qwen3-4B-GGUF`
**vLLM AWQ:** `Qwen/Qwen3-4B-AWQ`

---

### Gemma 3 (Google, March 2025)

**Overview:**
- **Developer**: Google DeepMind
- **Release**: March 2025
- **Architecture**: Dense transformer (1B text-only, 4B+ multimodal)

**Model Variants:**

| Model | Modality | Context | VRAM (Q4) |
|-------|----------|---------|-----------|
| 1B | Text only | 32K | ~0.8GB |
| 4B | Multimodal | 128K | ~2GB |
| 12B | Multimodal | 128K | ~6GB |
| 27B | Multimodal | 128K | ~14GB |

**Key Features:**
- **QAT (Quantization-Aware Training)**: Preserves quality at 4-bit
- **Multimodal** (4B+): Image understanding, visual reasoning
- **140+ languages**: Extensive multilingual support

**QAT Benefits:**
Google's QAT process trains the model with quantization in mind, so 4-bit Gemma 3 retains more quality than post-training quantized models.

**VRAM with QAT:**
| Model | Standard Q4 | QAT Q4 |
|-------|-------------|--------|
| 4B | ~2.5GB | ~2GB |

**Best For:**
- Multimodal tasks (image + text)
- Multilingual applications
- Quality-sensitive 4-bit deployments

**Ollama Tags:** `gemma3:1b`, `gemma3:4b`
**GGUF Source:** `google/gemma-3-*-it-gguf`

---

### Ministral 3B (Mistral, 2025)

**Overview:**
- **Developer**: Mistral AI
- **Release**: 2025
- **Architecture**: 3.4B language + 0.4B vision encoder

**Key Features:**
- **Edge-optimized**: Designed for on-device deployment
- **Function calling**: Native structured output support
- **Multimodal**: Vision encoder for image understanding
- **Agent-ready**: Built for autonomous task execution

**VRAM Requirements:**
| Quantization | VRAM | Notes |
|--------------|------|-------|
| Q4_K_M | ~2.5GB | Text + vision |
| FP16 | ~7GB | Not recommended for 6GB |

**Best For:**
- Edge deployment
- Agent workflows
- Function calling and tool use
- Compact multimodal applications

**Ollama Tag:** `ministral:3b`
**GGUF Source:** `mistralai/Ministral-3B-Instruct-GGUF`

---

## Recommended Test Order for 6GB VRAM

Based on our research, here's the recommended testing priority:

### Phase 1: Priority Models

1. **Qwen3-4B** - Proven performer, good ecosystem, start here
2. **SmolLM3-3B** - Compare dual-mode reasoning vs Qwen3
3. **Granite 4.0-H-Tiny** - Test hybrid architecture claims (should use least VRAM)
4. **Granite 4.0-H-Micro** - Dense hybrid comparison

### Phase 2: Strong Candidates

5. **Gemma 3 4B QAT** - Multimodal baseline
6. **Ministral 3B** - Edge/agent comparison
7. **Llama 3.2 3B** - Meta baseline

### Phase 3: Baselines

8. **Qwen3-1.7B** - Mid-size baseline
9. **Gemma 3 1B** - Smallest Gemma
10. **Qwen3-0.6b** - Smallest Qwen (speed baseline)

---

## VRAM Budget Analysis for 6GB GPU

### Available VRAM Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| Total VRAM | 6.0GB | GTX 1660 Super |
| CUDA overhead | 0.3-0.5GB | Driver, contexts |
| Display (if connected) | 0.2-0.5GB | Framebuffer |
| **Available for inference** | **5.0-5.5GB** | Best case |

### What Fits (Q4_K_M Quantization)

| Model Size | Model VRAM | KV Cache (4K) | Total | Fits? |
|------------|------------|---------------|-------|-------|
| 1B | 0.8GB | 0.5GB | 1.3GB | ✅ Yes |
| 3B | 2.0GB | 0.5GB | 2.5GB | ✅ Yes |
| 4B | 2.5GB | 0.5GB | 3.0GB | ✅ Yes |
| 7B | 4.0GB | 0.5GB | 4.5GB | ⚠️ Tight |
| 13B | 7.5GB | 0.5GB | 8.0GB | ❌ No |

### Engine Comparison

| Engine | VRAM Efficiency | Dynamic Loading | Best For |
|--------|-----------------|-----------------|----------|
| Ollama | Good | Yes | Interactive use |
| llama.cpp | Best | Manual | Production, max efficiency |
| vLLM | Worst | No | Batch processing |

**Note**: vLLM pre-allocates KV cache, so a 1B model can use 4+ GB if `max_model_len` is high.

---

## Architecture Innovations

### Mamba-2 (IBM Granite)

Traditional transformers have O(n²) memory scaling with sequence length due to attention. Mamba uses:
- **State Space Models (SSM)**: O(n) scaling
- **Selective scanning**: Content-aware state updates
- **Hardware-aware design**: Optimized for GPU memory hierarchy

**Result**: 70% less RAM for same capability.

### QAT (Google Gemma)

Standard quantization:
1. Train FP32/FP16 model
2. Convert to 4-bit post-training
3. Quality loss from approximation

QAT process:
1. Train with quantization simulation
2. Model learns to be robust to quantization
3. Final 4-bit model retains more quality

---

## Data Sources

- SmolLM3: https://huggingface.co/blog/smollm3
- Granite 4.0: https://www.ibm.com/granite/docs/models/granite
- Qwen3: https://qwenlm.github.io/blog/qwen3/
- Gemma 3: https://huggingface.co/blog/gemma3
- Ministral: https://mistral.ai/news/ministraux/

---

## Future Models to Watch

- **Llama 4** (Meta) - Expected Q1 2026
- **Claude Haiku 4** (Anthropic) - Potential API-only small model
- **Phi-5** (Microsoft) - Continuation of efficient model line
- **Granite 5** (IBM) - Further Mamba improvements

---

*Last updated: December 2025*
