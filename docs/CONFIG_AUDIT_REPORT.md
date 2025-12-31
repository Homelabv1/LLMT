# Model Configuration Audit Report

**Date**: December 2025
**Auditor**: Automated audit script
**Status**: Issues Found - Fixes Applied

## Audit Scope

This audit verifies that model configuration files follow the inheritance rule:
- **1x GPU models ⊆ 2x GPU models ⊆ 4x GPU models**
- Larger VRAM configurations must include ALL models from smaller configurations

## GPU Configuration Hierarchy

```
1660super-1x (6GB)
    └── 1660super-2x (12GB)
            └── 1660super-4x (24GB)

1660super-1x (6GB)
    └── 3060ti (8GB) ≈ 3070 (8GB)
            └── 3090ti (24GB)
                    └── 3090ti-3060ti (32GB)
```

---

## Ollama Configuration Issues

### 1660super-2x (12GB)

**Missing models from 1660super-1x:**
| Model | Priority | Notes |
|-------|----------|-------|
| smollm3:3b | Tier 1 | Dual-mode reasoning |
| granite4-tiny:latest | Tier 1 | Hybrid architecture |
| granite4-micro:latest | Tier 1 | Dense hybrid |
| ministral:3b | Tier 2 | Function calling |
| granite4-nano:1b | Tier 3 | Edge optimized |
| granite4-nano:350m | Tier 3 | Smallest |
| tinyllama:1.1b | Tier 3 | Legacy baseline |
| stablelm2:1.6b | Tier 3 | Legacy |
| deepseek-coder:1.3b | Tier 3 | Code model |

### 1660super-4x (24GB)

**Missing models from 1660super-2x:**
| Model | Notes |
|-------|-------|
| codellama:7b-instruct-q4_K_M | Present in 2x but missing in 4x |
| *All models missing from 2x* | Cascading inheritance issue |

### 3060ti (8GB) / 3070 (8GB)

**Missing models from 1660super-1x:**
| Model | Priority |
|-------|----------|
| smollm3:3b | Tier 1 |
| granite4-tiny:latest | Tier 1 |
| granite4-micro:latest | Tier 1 |
| ministral:3b | Tier 2 |
| granite4-nano:1b | Tier 3 |
| granite4-nano:350m | Tier 3 |
| tinyllama:1.1b | Tier 3 |
| stablelm2:1.6b | Tier 3 |
| deepseek-coder:1.3b | Tier 3 |

### 3090ti (24GB)

**Missing models:**
- codellama:7b-instruct-q4_K_M (should inherit from 8GB cards)
- All models missing from 3060ti/3070

### 3090ti-3060ti (32GB)

**Missing models:**
- deepseek-r1:14b (present in 4x but not in 3090ti config)
- codellama:13b-instruct-q4_K_M (present in 4x but not in 3090ti config)
- codellama:34b-instruct-q4_K_M (present in 3090ti but not in mixed config)
- All models missing from 3090ti

---

## llama.cpp Configuration Issues

### 1660super-2x (12GB)

**Missing models from 1660super-1x:**
| Model | Notes |
|-------|-------|
| /models/smollm3-3b-q4_K_M.gguf | Priority model |
| /models/granite-4.0-tiny-q4_K_M.gguf | Hybrid architecture |
| /models/granite-4.0-micro-q4_K_M.gguf | Dense hybrid |
| /models/qwen3-4b-q4_K_M.gguf | 256K context |
| /models/gemma-3-4b-it-q4_K_M.gguf | QAT optimized |
| /models/ministral-3b-instruct-q4_K_M.gguf | Function calling |
| /models/qwen3-1.7b-q4_K_M.gguf | Mid-size |
| /models/qwen3-0.6b-q4_K_M.gguf | Speed baseline |
| /models/granite-4.0-nano-1b-q4_K_M.gguf | Edge optimized |
| /models/tinyllama-1.1b-chat-v1.0-q4_K_M.gguf | Legacy |

**Extra model not in 1x (keep):**
- /models/qwen2.5-3b-instruct-q4_K_M.gguf

### 1660super-4x (24GB)

**Missing models from 1660super-2x:**
| Model | Notes |
|-------|-------|
| /models/qwen2.5-3b-instruct-q4_K_M.gguf | Present in 2x |
| /models/llama-3.2-3b-instruct-q4_K_M.gguf | Present in 2x |
| /models/phi-4-mini-instruct-q4_K_M.gguf | Present in 2x |
| /models/gemma-2-2b-it-q4_K_M.gguf | Present in 2x |
| /models/codellama-7b-instruct-q4_K_M.gguf | Present in 2x |
| *All models missing from 2x* | Cascading |

### 3060ti (8GB)

**Missing models from 1660super-1x:**
- All new Tier 1 models (smollm3, granite4, etc.)
- gemma-2-2b missing but gemma-3-4b should be there

### 3090ti (24GB) / 3090ti-3060ti (32GB)

**Missing many smaller models** - same pattern as Ollama

---

## vLLM Configuration Issues

### 1660super-2x (12GB)

**Missing models from 1660super-1x:**
| Model | Notes |
|-------|-------|
| Qwen/Qwen2.5-3B-Instruct-AWQ | AWQ quantized |
| Qwen/Qwen3-4B-AWQ | AWQ quantized |
| google/gemma-3-4b-it-gptq-int4 | GPTQ quantized |
| microsoft/Phi-3-mini-4k-instruct-awq | AWQ quantized |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | FP16 small |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | FP16 small |

### 1660super-4x (24GB)

**Missing models from 1660super-2x:**
| Model | Notes |
|-------|-------|
| Qwen/Qwen2.5-1.5B-Instruct | FP16 |
| meta-llama/Llama-3.2-1B-Instruct | FP16 |
| TheBloke/Mistral-7B-Instruct-v0.2-AWQ | AWQ 7B |
| TheBloke/Llama-2-7B-Chat-AWQ | AWQ 7B |
| *All models missing from 2x* | Cascading |

### 3060ti (8GB) / 3090ti (24GB) / 3090ti-3060ti (32GB)

**Similar inheritance issues** - missing quantized models from smaller configs

---

## Summary

| Engine | Configs Audited | Issues Found | Severity |
|--------|-----------------|--------------|----------|
| Ollama | 7 | 6 configs with missing models | High |
| llama.cpp | 7 | 6 configs with missing models | High |
| vLLM | 7 | 5 configs with missing models | High |

**Total inheritance violations**: 45+ missing models across all configs

---

## Fixes Applied

All configuration files have been updated to ensure proper inheritance:

1. **1660super-2x**: Added all models from 1x
2. **1660super-4x**: Added all models from 2x
3. **3060ti/3070**: Added new Tier 1 models from research
4. **3090ti**: Added all models from 8GB cards
5. **3090ti-3060ti**: Added all models from 3090ti

Each config now follows the format:
```
# [GPU Config Name]
# [Description]

# =============================================================================
# INHERITED FROM [smaller config]
# =============================================================================
[models from smaller config]

# =============================================================================
# NEW MODELS FOR [this VRAM tier]
# =============================================================================
[additional models for this tier]
```

---

*Report generated: December 2025*
