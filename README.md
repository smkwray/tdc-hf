# TDC-HF

High-frequency tooling for Treasury Deposit Component (TDC) measurement and
diagnostics.

## About

TDC-HF extends quarterly Treasury Deposit Component estimates to higher
frequency measurement and diagnostic workflows. It provides reproducible data
assembly, temporal benchmarking, local-projection analysis, IV diagnostics, and
closeout checks for studying monthly Treasury, banking, and money-market
channels.

This repository builds monthly and weekly datasets from public Treasury,
Federal Reserve, FRED, FiscalData, TIC, and auction-context sources. The main
monthly proxy is benchmarked to quarterly TDC estimates and is used for
high-frequency local-projection, IV, robustness, and closeout diagnostics.

GitHub: [`smkwray/tdc-hf`](https://github.com/smkwray/tdc-hf).

## Related Projects

- [`tdcest`](https://github.com/smkwray/tdcest): quarterly TDC estimates and accounting definitions.
- [`tdcpass`](https://github.com/smkwray/tdcpass): lower-frequency pass-through analysis.
- [`tsyparty`](https://github.com/smkwray/tsyparty): Treasury auction and holder-sector context.
- [`wamest`](https://github.com/smkwray/wamest): holder maturity and sector context.
- [`tdcsfc`](https://github.com/smkwray/tdcsfc): accounting and scenario references.

## Outputs

The package can generate:

- monthly TDC proxy components benchmarked to quarterly anchors;
- monthly proxy validation and method-comparison tables;
- weekly Treasury-plumbing state panels;
- TGA, fiscal-flow, category-flow, and auction shock measures;
- LP, LP-IV, bootstrap, placebo, regime, and factor-control diagnostics;
- closeout tables comparing IV, non-IV, and reduced-form designs.

Generated data and outputs are ignored by git. The tracked files are the Python
package, analysis specs, tests, and documentation needed to reproduce them.

## Setup

```bash
set -a
source .env
set +a
uv pip install -e '.[dev]'
```

Run tests:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m pytest
```

## Common Commands

Run the full Denton/FRED/DTS analysis spec:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdchf run-analysis-spec \
  config/analysis_fred_denton_shock.yml
```

Run selected closeout artifacts:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -B -m tdchf run-analysis-spec \
  config/analysis_fred_denton_shock.yml \
  --only closeout_anchor_contract_audit,closeout_short_run_core_lp_iv,closeout_noniv_tdc_lp,closeout_tga_reduced_form,closeout_placebo_summary,closeout_method_envelope,closeout_core_iv_vs_noniv_profiles
```

Run a minimal demo:

```bash
"$UV_PROJECT_ENVIRONMENT/bin/tdchf" demo --out output/demo
```

## Main Specs

- `config/analysis_fred_denton_shock.yml`: current Denton benchmarked
  high-frequency analysis.
- `config/analysis_fred_shock.yml`: residual-spread comparison spec.
- `config/series_registry.yml`: public series registry.
- `config/upstream_sources.yml`: expected sibling-project outputs.

## Repository Layout

```text
config/                 analysis specs and source contracts
data/raw/               local raw downloads, ignored
data/processed/         generated processed data and readouts, ignored
docs/                   local method and result notes, ignored
output/                 generated tables and figures, ignored
src/tdchf/              Python package
tests/                  regression tests
```

Method and result notes are generated locally under ignored directories; the
generated readouts under `data/processed/` are the primary result documents
after a local pipeline run.
