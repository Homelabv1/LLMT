# 1B Specialist Models - GTX 1660 Super

## Purpose

Test sub-2B models to prove fine-tuned small models can achieve 95%+ accuracy on bounded use cases.

## Why This Matters

| Factor | 32B Generalist | 1B Specialist |
|--------|----------------|---------------|
| VRAM | 18GB | 1GB |
| Speed | 10-20 tok/s | 60-100 tok/s |
| Accuracy (domain) | 85% general | 95%+ trained |
| Hardware cost | $1,299+ GPU | $150 used 1660S |
| Power | 250-300W | 75-100W |

## Usage

Uses existing framework - just point to this config:

```bash
# Test all models with Ollama
python scripts/batch_test.py --engine ollama --config gpus/nvidia/1660super-1b/models.yaml

# Test specific tier
python scripts/batch_test.py --engine ollama --config gpus/nvidia/1660super-1b/models.yaml --tier primary

# Resume interrupted test
python scripts/batch_test.py --engine ollama --config gpus/nvidia/1660super-1b/models.yaml --resume

# Check status
python scripts/batch_test.py --engine ollama --config gpus/nvidia/1660super-1b/models.yaml --status

# Dry run
python scripts/batch_test.py --engine ollama --config gpus/nvidia/1660super-1b/models.yaml --dry-run
```

## Models (10 total)

| Tier | Models | Test Order |
|------|--------|------------|
| Primary | Qwen2.5-1.5B, Qwen3-1.7B, SmolLM2-1.7B | First |
| Secondary | Llama-3.2-1B, DeepSeek-R1-Distill-1.5B | If primary insufficient |
| Baseline | Qwen2.5-0.5B, TinyLlama-1.1B | Lower bound |
| Optional | StableLM-2, Phi-1.5, Qwen2.5-Coder | If time permits |

## VRAM Estimates

| Size | VRAM (Q4) | Headroom on 6GB |
|------|-----------|-----------------|
| 0.5B | ~0.5GB | 5.5GB free |
| 1.1B | ~0.8GB | 5.2GB free |
| 1.5B | ~1.0GB | 5.0GB free |
| 1.7B | ~1.2GB | 4.8GB free |

## Results Location

Results go to: `gpus/nvidia/1660super-1b/results/<engine>/<model>/`
