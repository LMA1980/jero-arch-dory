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

______________________________________________________________________

NVIDIA Nemotron-Labs Dory1.0 looped latent-recurrent normalized-GPT (`nGPT`) architecture recipe (`dory`) for the SpacemiT K3 platform, supporting 3-segment looped execution (`n_input_layers`, `n_recurrent_layers` × `n_loops`, `n_output_layers`), interleaved sliding-window (`S`) / full (`*`) attention and MLP (`-`) sub-layers, and `per_loop` / `last_loop` KV cache modes.

## Ecosystem & Documentation

- **Public Documentation Portal**: [`lma1980.github.io/model-runner`](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner) ([Reference Manual](https://github.com/LMA1980/lma1980.github.io/blob/main/model-runner/manual.md), [Crate Docs](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner/docs/crates))
- **Upstream Dependencies**: [`jero-core`](https://github.com/LMA1980/jero-core) and [`jero-kernels-rvv`](https://github.com/LMA1980/jero-kernels-rvv) (via the optional `with-kernels` feature)
- **Downstream & Companion Crates**: [`jero-engine`](https://github.com/LMA1980/jero-engine) and [`jero-kv-cache`](https://github.com/LMA1980/jero-kv-cache)

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
