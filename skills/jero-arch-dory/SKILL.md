---
name: jero-arch-dory
description: "Implement and validate NVIDIA Nemotron-Labs Dory1.0 (dory) looped latent-recurrent nGPT ArchRecipe execution, hyperspherical L2 normalization, hybrid SWA/Full attention, and per_loop/last_loop KV cache routing in jero-arch-dory."
---

# `jero-arch-dory` Architecture Recipe Invariants

- Implement `jero_core::ArchRecipe` for the Dory1.0 architecture (`dory`), partitioning physical sub-layers into `n_input_layers`, `n_recurrent_layers` (executed for $K \in \{1..=4\}$ loops with loop-entry `L2Norm(cur + input_h)` re-injection and loop-exit `L2Norm(cur)`), and `n_output_layers`.
- Enforce hyperspherical Normalized-GPT (`nGPT`) semantics: parameter-free unit L2 normalization (`eps = 1e-6`), learned QK head scaling (`attn_qk_scale`), sigmoid attention gating (`attn_gate` + `attn_gate_scale`), scaled SwiGLU MLP (`ffn_gate_scale` + `ffn_up_scale`), and eigen-rate residual interpolation (`residual_alpha`).
- Strictly honor the GGUF `dory.recurrent_kv_cache_mode` (`"per_loop"` vs `"last_loop"`) unless explicitly overridden via CLI or `TuningProfile`.
- Use `usize` for tensor slice indexing and memory offsets, and `u32` for architectural hyperparameters (`n_input_layers`, `n_recurrent_layers`, `n_output_layers`, `n_loops`, `sliding_window`).
