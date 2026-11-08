# jero-arch-dory

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/jero-arch-dory/card.svg" alt="jero-arch-dory Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-dory/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-dory/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-dory/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-dory/benchmark.svg" alt="Benchmark"/></a>
</p>

---

NVIDIA Nemotron-Labs Dory1.0 looped latent-recurrent normalized-GPT (`nGPT`) architecture recipe (`dory`) for the SpacemiT K3 platform, supporting 3-segment looped execution (`n_input_layers`, `n_recurrent_layers` × `n_loops`, `n_output_layers`), interleaved sliding-window (`S`) / full (`*`) attention and MLP (`-`) sub-layers, and `per_loop` / `last_loop` KV cache modes.

## Architecture & Specifications
For master architecture and detailed component specifications, see the parent [model-runner](https://github.com/LMA1980/model-runner) repository:
- Master Plan: [PLAN.md](https://github.com/LMA1980/model-runner/blob/main/PLAN.md)
- Specs: [specs/runtime/](https://github.com/LMA1980/model-runner/tree/main/specs/runtime)

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
