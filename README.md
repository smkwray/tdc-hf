# TDC-HF

High-frequency tooling for Treasury Deposit Component (TDC) measurement and
diagnostics.

## About

TDC-HF combines quarterly Treasury Deposit Component estimates with
higher-frequency measurement and diagnostic workflows. It provides
reproducible data assembly, temporal benchmarking, local-projection analysis,
IV diagnostics, and closeout checks for studying monthly Treasury, banking, and
money-market channels.

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
- a deterministic daily/weekly/quarterly Treasury-flow bridge;
- unnormalized DTS–H.8 cumulative deposit-stock response profiles;
- TGA, fiscal-flow, category-flow, and auction shock measures;
- LP, LP-IV, bootstrap, placebo, regime, and factor-control diagnostics;
- closeout tables comparing IV, non-IV, and reduced-form designs.

Generated data and outputs are ignored by git. The tracked files are the Python
package, analysis specs, tests, and documentation needed to reproduce them.

## Setup

```bash
uv venv
uv pip install -e '.[dev]'
```

Run tests:

```bash
uv run --no-sync python -B -m pytest
```

## Common Commands

Run the full Denton/FRED/DTS analysis spec:

```bash
uv run --no-sync python -B -m tdchf run-analysis-spec \
  config/analysis_fred_denton_shock.yml
```

Run selected closeout artifacts:

```bash
uv run --no-sync python -B -m tdchf run-analysis-spec \
  config/analysis_fred_denton_shock.yml \
  --only closeout_anchor_contract_audit,closeout_short_run_core_lp_iv,closeout_noniv_tdc_lp,closeout_tga_reduced_form,closeout_placebo_summary,closeout_method_envelope,closeout_core_iv_vs_noniv_profiles
```

Run a minimal demo:

```bash
uv run --no-sync tdchf demo --out output/demo
```

## Deterministic Quarterly/High-Frequency Bridge

Build the sparse bridge:

```bash
uv run --no-sync python -B -m tdchf build-quarterly-hf-bridge \
  --out output/tables/quarterly_hf_bridge.csv
```

The ignored CSV has 52 columns and five row types: `event_daily`,
`h8_reconciliation`, `quarter_reconciliation`, `input_coverage`, and
`quarter_anchor`. Observed events use the DTS settlement `record_date` for
calendar-quarter assignment. They also retain the separate Thursday-through-
Wednesday H.8 window ending Wednesday and a calendar-day quarter-end age bin
(`0-7`, `8-30`, `31-60`, or `61+`).

The observed lane keeps six source-signed streams separate: gross credits,
gross drains, gross debt issuance, gross debt redemption, broad-outflow
sidecar, and interest-outflow sidecar. It reconciles each stream independently
at complete H.8 weeks and settlement-date quarters. Its outcome perimeter is
H.8 commercial-bank deposits. The bridge is not a weekly TDC level, does not
net observed flows into quarterly TDC, and does not refit a timed estimator.

Quarterly comparison rows are distinct from observed timing. They contain the
current Tier-2 headline
`tdc_tier2_interest_corrected_bank_only_ru_flow`, the current
`tdc_base_bank_only_ru_flow` comparison, and the legacy five-component Denton
quarterly identity. The current comparisons have no intraquarter timing and no
observed-versus-anchor accounting residual; the legacy comparison is labeled
generated rather than observed.

In addition to the project-local DTS, weekly, crosswalk, and validated legacy
inputs, the command consumes these sibling `tdcest` outputs:

- `../tdcest/data/processed/tdc_estimates.csv`
- `../tdcest/data/processed/tdc_downstream_handoff_bundle.json`

The estimates file supplies quarter-end values in USD millions for the exact
Tier-2 and base columns above; the bridge scales them to USD billions without
filling, extrapolating, or allocating them within the quarter. The handoff file
must designate the Tier-2 series as the current corrected headline and retain
the base series as a comparison. This current comparison contract is separate
from the legacy five-component Denton contract in
`config/upstream_sources.yml`.

## Directly Timed DTS–H.8 Response Profiles

Build the descriptive response-profile artifacts:

```bash
uv run --no-sync python -B -m tdchf run-disbursement-response-profile
```

`run-disbursement-response-profile` estimates conditional associations between
gross Daily Treasury Statement flow buckets and cumulative changes in the H.8
“Deposits, All Commercial Banks” stock from the preceding Wednesday through
each horizon. Coefficients are reported in billions of H.8 deposits per billion
dollars of the named gross DTS flow.

The public output preserves the descriptive signal and its pointwise
uncertainty. If the DTS-to-H.8 perimeter mapping is stable and conditioning on
realized future flows adequately handles anticipation, positive later
coefficients are consistent with a delayed measured-perimeter deposit response.
The future-flow-conditioned LP and the finite distributed-lag model remain
distinct specifications, so their contrasts are reported rather than averaged
away.

The DTS treatment is not matched to beneficiary deposit accounts or to the H.8
reporting perimeter. The same-week coefficient is therefore a measured-
perimeter impact response, not a landing share. Later coefficients do not
separately identify immediate incidence, a post-landing kernel, retention,
causal decay, or a deposit half-time. The output contains no horizon-to-impact
normalization or state-capacity estimate. State, placebo, and stability
failures are retained only in the diagnostic `state_response_gate_status.csv`
admission receipt; they are not a normal command-line workflow.

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
