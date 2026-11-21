# tdc-hf

`tdc-hf` builds daily, weekly, and monthly Treasury-flow panels and estimates high-frequency deposit and reserve projections around Treasury cash flows, H.4.1 reserve factors, and Quarterly Refunding Announcements (QRA). It reads public Federal Reserve and Treasury data alongside quarterly and weekly tables from [tdcest](https://github.com/smkwray/tdcest) and writes CSV tables, Markdown readouts, and JSON receipts under `output/`.

## What it does

The `tdchf` CLI groups four analysis workflows:

- `build-quarterly-hf-bridge` — reconciles Daily Treasury Statement (DTS) gross flows (credits, drains, debt issuance, and debt redemption) by settlement quarter and H.8 bank week (Thursday through Wednesday, labeled by the ending Wednesday) against quarterly `tdcest` series.
- `run-disbursement-response-profile` — estimates cumulative H.8 commercial-bank deposit-stock projections on gross DTS flows across leads and lags.
- `run-weekly-reserve-offsets`, `run-weekly-trc-projection`, `run-weekly-reserve-history`, `run-weekly-reserve-rolling`, and `run-monthly-reserve-rolling` — estimate weekly reserve offsets on Treasury cash legs (`-d(TGA) - d(SFP)`), weekly projections on the core Treasury component of reserves (TRC) with period-specific restricted-score wild bootstrap inference, and rolling weekly and monthly reserve paths (adding the money-fund route on the monthly last-Wednesday clock).
- `run-analysis-spec`, `build-qra-borrowing-surprise`, and `run-qra-event-lp` — allocate quarterly TDC anchors to monthly series via YAML workflows (`config/analysis_fred_denton_shock.yml`, `config/analysis_fred_shock.yml`) and estimate QRA borrowing-surprise local projections.

## Data

All external series are public:

- Federal Reserve FRED series (H.8 commercial-bank deposits, H.4.1 reserve factors, and macro controls declared in `config/series_registry.yml`), downloaded from FRED's public `fredgraph.csv` endpoint without an API key.
- Treasury Fiscal Data API (`api.fiscaldata.treasury.gov`), covering DTS operating cash balances, deposits and withdrawals of operating cash, income tax refunds issued, and federal tax deposits.
- Upstream quarterly TDC tables (`tdc_estimates.csv` and summary metadata) and hash-pinned H.4.1 and weekly TRC exports from `tdcest`. By default, `config/upstream_sources.yml` looks for a sibling checkout at `../tdcest`; individual commands accept path overrides such as `--source-root`, `--tdc-estimates`, or `--ledger`.

## Install

Requires Python 3.11 or later. From the repository root:

```bash
uv venv
uv pip install -e '.[dev]'
```

## Usage

Check the CLI and run the offline synthetic benchmarking demo (writes `tdc_monthly_components.csv`, `tdc_monthly_proxy.csv`, and `tdc_validation_report.json` to `output/demo/`):

```bash
uv run --no-sync python -B -m tdchf --help
uv run --no-sync python -B -m tdchf demo --out output/demo
```

Check whether sibling upstream files declared in `config/upstream_sources.yml` are present:

```bash
uv run --no-sync python -B -m tdchf doctor
```

Inspect required paths and SHA-256 options for any production command with `--help`:

```bash
uv run --no-sync python -B -m tdchf build-quarterly-hf-bridge --help
uv run --no-sync python -B -m tdchf run-disbursement-response-profile --help
uv run --no-sync python -B -m tdchf run-weekly-reserve-offsets --help
uv run --no-sync python -B -m tdchf run-weekly-trc-projection --help
uv run --no-sync python -B -m tdchf run-weekly-reserve-history --help
uv run --no-sync python -B -m tdchf run-weekly-reserve-rolling --help
uv run --no-sync python -B -m tdchf run-monthly-reserve-rolling --help
uv run --no-sync python -B -m tdchf run-analysis-spec --help
```

Pinned reserve commands read their config JSON files under `config/` (for example `config/weekly_trc_projection_v1.json`) and stop if an input file is missing or fails SHA-256 verification.

## Output

Generated CSV tables, Markdown readouts, manifests, and JSON provenance receipts are written to ignored directories under `output/`—including synthetic monthly demo tables under `output/demo/`, disbursement deposit-stock profile tables, and weekly and monthly reserve projection tables.

## Limits

- Gross DTS disbursement and tax-drain totals aggregate all Treasury cash flows without matching beneficiaries to commercial-bank accounts, so H.8 deposit-stock slopes are unnormalized conditional associations (billions of deposits per billion dollars of flow).
- Monthly temporal allocations interpolate quarterly anchors; intraquarter timing comes from the monthly indicator series.
- The H.4.1 reverse-repurchase "other counterparties" series mixes several counterparty types and provides only a coarse proxy for ON RRP or money-fund settlement routes.
- Integration tests and production commands require local DTS, QRA, and `tdcest` inputs that are excluded from the repository.

## Tests

Run the offline fixture-based test suite and linter:

```bash
uv run --no-sync python -B -m pytest
uvx ruff check .
```

The default pytest configuration excludes tests marked `integration` (which check local DTS, QRA, and state-response datasets). Run them when local data are present:

```bash
uv run --no-sync python -B -m pytest -m integration
```
