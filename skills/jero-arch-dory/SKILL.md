---
name: jero-arch-dory
description: NVIDIA Nemotron-Labs Dory1.0 looped latent-recurrent nGPT architecture recipe, physical sub-layer weight/scale cache, and ChatML + think metadata for K3 inference.
---

# `jero-arch-dory` Architecture Crate

Provides the standalone `DoryRecipe`, `DoryPhysicalCache`, and `DoryStopMetadata` implementations for the NVIDIA Nemotron-Labs Dory1.0-4B looped latent-recurrent normalized GPT (`nGPT`) architecture (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`).

## Key Capabilities

- **GGUF & Metadata Hydration**: `DoryRecipe::from_gguf` and `DoryRecipe::from_metadata` parse `dory.layer_pattern`, `n_loops`, `recurrent_kv_cache_mode` (`per_loop` / `last_loop`), dual-theta NeoX `RoPE` (`theta_1 = 10_000_000`, `theta_2 = 10_000`), `sliding_window`, `init_std`, `alpha_init`, `gate_init`, and `logit_scale`.
- **Model-Scoped Physical Cache (`DoryPhysicalCache`)**: Repacks 2D weights once at load time strictly across physical sub-layers (`0..physical_layer_count()` — never duplicated per virtual loop iteration) into `128-byte`-aligned `DoryCachedWeight` (`jero-kernels-rvv::RepackedWeight` under `with-kernels`, aligned scalar storage under `--no-default-features`) across `Q5_K`, `Q6_K`, `Q8_0`, `Q4_K_M`, `Q4_0`, `Q5_0`, `F32`, `F16`, and `BF16`.
- **Pre-Decoded `128-Byte`-Aligned `nGPT` Scale Vectors (`AlignedF32Vec`)**: Decodes `sqk`, `suv`, `residual_alpha`, `attn_gate_scale`, `ffn_gate_scale`, and optional `output_scale.weight` (`F32`/`F16`/`BF16`) once at load time, eliminating per-token `extract_scale_vector_f32` overhead.
- **`ChatML` + `<think>` Alignment (`DoryStopMetadata`)**: Resolves `<s>`, `<|im_start|>`, `<|im_end|>`, `<think>`, and `</think>` token IDs from GGUF tokenizer metadata, excluding reasoning boundary tags from `StopCriteria` while halting on `<|im_end|>` / EOS.
