# Model Configuration Audit Report

**Date**: December 2025
**Status**: ✅ All Issues Resolved

## Audit Scope

This audit verifies that model configuration files follow inheritance rules:
- **Within-GPU Multiplicity**: 1x ⊆ 2x ⊆ 4x
- **Cross-GPU VRAM**: Smaller VRAM configs ⊆ Larger VRAM configs

## GPU Configuration Hierarchy

```
Within-GPU Multiplicity (1660 Super):
    1660super-1x (6GB)
        └── 1660super-2x (12GB)
                └── 1660super-4x (24GB)

Cross-GPU VRAM Inheritance:
    1660super-1x (6GB)
        └── 3060ti (8GB) ≈ 3070 (8GB)
                └── 3090ti (24GB)
                        └── 3090ti-3060ti (32GB)
```

---

## Audit Results by Engine

### Ollama

| Configuration | Status | Models | Inherits From |
|---------------|--------|--------|---------------|
| 1660super-1x | ✅ | 20 | Base config |
| 1660super-2x | ✅ | 26 | 1660super-1x |
| 1660super-4x | ✅ | 33 | 1660super-2x |
| 3060ti | ✅ | 26 | 1660super-1x |
| 3070 | ✅ | 26 | 1660super-1x (same VRAM as 3060ti) |
| 3090ti | ✅ | 33 | 3060ti/3070 |
| 3090ti-3060ti | ✅ | 39 | 3090ti |

### llama.cpp (GGUF)

| Configuration | Status | Models | Inherits From |
|---------------|--------|--------|---------------|
| 1660super-1x | ✅ | 13 | Base config |
| 1660super-2x | ✅ | 19 | 1660super-1x |
| 1660super-4x | ✅ | 27 | 1660super-2x |
| 3060ti | ✅ | 19 | 1660super-1x |
| 3070 | ✅ | 19 | 1660super-1x (same VRAM as 3060ti) |
| 3090ti | ✅ | 28 | 3060ti/3070 |
| 3090ti-3060ti | ✅ | 31 | 3090ti |

### vLLM (HuggingFace)

| Configuration | Status | Models | Inherits From |
|---------------|--------|--------|---------------|
| 1660super-1x | ✅ | 8 | Base config |
| 1660super-2x | ✅ | 14 | 1660super-1x |
| 1660super-4x | ✅ | 22 | 1660super-2x |
| 3060ti | ✅ | 14 | 1660super-1x |
| 3070 | ✅ | 14 | 1660super-1x (same VRAM as 3060ti) |
| 3090ti | ✅ | 22 | 3060ti/3070 |
| 3090ti-3060ti | ✅ | 26 | 3090ti |

---

## Summary

| Engine | Total Configs | Passing | Issues |
|--------|---------------|---------|--------|
| Ollama | 7 | 7 | 0 |
| llama.cpp | 7 | 7 | 0 |
| vLLM | 7 | 7 | 0 |
| **Total** | **21** | **21** | **0** |

---

## Fixes Applied (This Session)

### Created Missing RTX 3070 Config Files

The RTX 3070 (8GB) was missing llama.cpp and vLLM config files. Created:

1. `gpus/nvidia/3070/configs/llamacpp_models.txt`
   - Inherits all 19 models from 3060ti (same 8GB VRAM)
   - Includes Tier 1, 2, 3 models from 1660super-1x
   - Includes 7B Q4 quantized models

2. `gpus/nvidia/3070/configs/vllm_models.txt`
   - Inherits all 14 models from 3060ti (same 8GB VRAM)
   - Includes AWQ/GPTQ quantized models
   - Includes FP16 small models

### Previous Session Fixes

1. **Within-GPU Multiplicity** (1660 Super 1x → 2x → 4x):
   - Added all missing Tier 1 models (SmolLM3, Granite 4.0, Qwen3, Gemma 3)
   - Added all missing Tier 2 models (Ministral, Phi-4 Mini)
   - Added all missing Tier 3 models (Qwen3-0.6B, Granite Nano, TinyLlama)

2. **Cross-GPU VRAM Inheritance**:
   - 3060ti/3070 now includes all 1660super-1x models plus 7B variants
   - 3090ti now includes all 8GB card models plus 13-14B+ models
   - 3090ti-3060ti now includes all 3090ti models plus 70B models

---

## Configuration File Format

All config files now follow this standardized format:

```
# [Engine] models for [GPU] ([VRAM] VRAM)
# [Engine-specific notes]
# Updated: [Date]

# =============================================================================
# INHERITED FROM [parent config] - ALL MODELS MUST BE INCLUDED
# =============================================================================

# --- From [smaller config] ---
[Tier 1 models]
[Tier 2 models]
[Tier 3 models]

# --- From [intermediate config] ---
[intermediate tier models]

# =============================================================================
# NEW MODELS FOR [VRAM] VRAM
# =============================================================================
[models unique to this tier]
```

---

## Inheritance Rules Reference

### Within-GPU Multiplicity
- 2x config MUST include ALL models from 1x
- 4x config MUST include ALL models from 2x

### Cross-GPU VRAM
- 8GB cards (3060ti, 3070) MUST include all 6GB (1660super-1x) models
- 24GB cards (3090ti) MUST include all 8GB card models
- 32GB configs (3090ti+3060ti) MUST include all 24GB models

### Same-VRAM GPUs
- GPUs with same VRAM (3060ti ≈ 3070) should have identical model lists
- Performance differences are noted in comments (3070 is ~10-15% faster)

---

*Audit completed: December 2025*
