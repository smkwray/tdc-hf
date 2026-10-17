# TDC-HF

TDC-HF builds high-frequency Treasury Deposit Component (TDC) measurement and
diagnostic datasets from public Treasury and Federal Reserve sources. It keeps
directly observed settlement flows, generated monthly allocations, and
announcement-time measures separate, and estimates descriptive deposit and
reserve associations with explicit timing, perimeter, and inference limits.

## What it provides

| Lane | Output and interpretation |
| --- | --- |
| Daily Treasury Statement bridge | Separate gross credits, drains, debt issuance, and debt redemption, reconciled by settlement quarter and by the Federal Reserve H.8 commercial-bank reporting week |
| Directly timed deposit profiles | Conditional associations between gross Treasury flows and cumulative changes in H.8 commercial-bank deposits |
| Weekly reserve projections | Treasury cash-leg or core Treasury Reserve Component (TRC) exposures, signed H.4.1 reserve-factor decompositions, event paths, and calibration diagnostics |
| Reserve history and rolling projections | Weekly period coefficients, two-year rolling weekly coefficients, and monthly/three-month reserve associations with a money-market-fund route term |
| Monthly benchmarking | Generated temporal allocations consistent with a specified quarterly anchor; comparison and validation tools |
| Announcement diagnostics | Quarterly Refunding Announcement (QRA) news measures, local projections, and instrumental-variable diagnostics |

A weekly channel dataset is not a weekly TDC level. Generated monthly allocations
do not observe intraquarter Treasury settlement timing. The legacy five-component
monthly benchmark does not allocate the current corrected quarterly TDC measure.

## Evidence limits

The directly timed deposit outcome is the H.8 commercial-bank deposit stock.
The Treasury treatments are gross Daily Treasury Statement (DTS) flow buckets,
which are not matched to beneficiary accounts or to that reporting perimeter.
Coefficients are unnormalized conditional associations, in billions of deposits
per billion dollars of the named flow. They do not separately identify immediate
incidence, a post-landing response, retention, causal decay, or a half-life.

State diagnostics retain coverage and failed admission checks. No state measure
is admitted as a capacity index or scenario selector. Weekly reserve calibration
failures remain visible in every affected coefficient export; a descriptive
coefficient does not become inferentially valid because its accounting
components reconcile. The reserve-history test inversion has its own declared
calibration and numerical resolution, which do not validate the other designs.
Rolling weekly and monthly projections are descriptive.

The reverse-repurchase-agreement (RRP) “other counterparties” balance used in the
reserve ledger mixes counterparties. It is neither an overnight RRP measure nor
a measure of money-fund funding routes. Reserve balances plus that balance is a
separate comparison outcome, not a reserve perimeter.

## Install and validate

Use Python 3.11 or later and run from a source checkout:

```bash
uv venv
uv pip install -e '.[dev]'
uv run --no-sync python -B -m tdchf --help
uv run --no-sync python -B -m pytest
uvx ruff check .
```

Run the self-contained synthetic benchmarking demo:

```bash
uv run --no-sync python -B -m tdchf demo --out output/demo
```

The demo uses six synthetic months and two quarterly anchors, and writes
monthly series plus a validation report. It needs no downloaded data or sibling checkout. The default test suite uses small fixtures and excludes the four data-backed
integration checks. Run those checks explicitly after supplying the required
local DTS, QRA, and state-response inputs:

```bash
uv run --no-sync python -B -m pytest -m integration
```

The integration lane fails when required data is absent; it does not turn a
missing-data check into a passing result.

## Data and reproducibility

[TDCest](https://github.com/smkwray/tdcest) supplies quarterly anchors and the
hash-pinned Treasury-flow and reserve-factor exports. Source locations and
contracts live in `config/upstream_sources.yml`, `config/series_registry.yml`,
and the versioned JSON designs. The default layout places `tdcest` next to this
checkout; the relevant commands expose source-path overrides.

The settlement bridge consumes validated local DTS inputs plus TDCest's
`data/processed/tdc_estimates.csv` and
`data/processed/tdc_downstream_handoff_bundle.json`. Its current comparisons
contain the exact `tdc_tier2_interest_corrected_bank_only_ru_flow` and
`tdc_base_bank_only_ru_flow` fields. Those producer identifiers are preserved;
the settlement-system complement is not a bank-sector classification.

Observed events use the DTS settlement date for calendar-quarter assignment.
H.8 windows run Thursday through Wednesday and are labeled by their ending
Wednesday. Each gross flow stream reconciles independently. Quarterly anchor
comparisons have no observed intraquarter timing and no accounting residual
forcing observed flows to equal a net TDC anchor.

Production commands verify their declared inputs and write local receipts and
artifacts. Inspect the command and its configuration before estimating:

```bash
uv run --no-sync python -B -m tdchf doctor
uv run --no-sync python -B -m tdchf build-quarterly-hf-bridge --help
uv run --no-sync python -B -m tdchf run-disbursement-response-profile --help
uv run --no-sync python -B -m tdchf run-weekly-reserve-offsets --help
uv run --no-sync python -B -m tdchf run-weekly-trc-projection --help
uv run --no-sync python -B -m tdchf run-weekly-reserve-history --help
uv run --no-sync python -B -m tdchf run-weekly-reserve-rolling --help
uv run --no-sync python -B -m tdchf run-monthly-reserve-rolling --help
```

`doctor` reports available sibling inputs; it is not a scientific approval.
A clean checkout does not contain production data. Hash mismatches and missing
required observations stop the relevant producer. Calibration runs can take
substantially longer than the synthetic demo or tests. Use a new output
location where a producer requires an absent destination; preserve existing
bundles and their receipts.

`run-analysis-spec` executes the ordered steps in a YAML specification.
`config/analysis_fred_denton_shock.yml` describes the generated monthly
benchmark and its diagnostics; `config/analysis_fred_shock.yml` describes the
residual-spread comparison. These workflows download public sources and require
the declared upstream contracts. They are not installation smoke tests.

## Repository layout

- `src/tdchf/`: acquisition, measurement, estimation, and reporting code.
- `config/`: analysis specifications, source contracts, and dated event definitions.
- `tests/`: fixture-based regression tests and the separate data-backed check.
- `data/`: raw downloads and processed panels, ignored by Git.
- `output/`: generated tables, figures, and receipts, ignored by Git.
- `docs/`: local method and result notes, ignored by Git.

Related projects include [TDCpass](https://github.com/smkwray/tdcpass) for
accounting boundary comparisons, [Tsyparty](https://github.com/smkwray/tsyparty)
for auction and holder-sector context, and
[WAMEST](https://github.com/smkwray/wamest) for maturity and sector context.
