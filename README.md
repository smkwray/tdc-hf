# tdc-hf

`tdc-hf` builds daily, weekly and monthly panels of US Treasury cash flows and estimates how commercial-bank deposits and bank reserves move around them. It uses the Daily Treasury Statement (DTS), the Federal Reserve's weekly H.4.1 release on factors affecting reserve balances, its H.8 release on bank balance sheets, and Quarterly Refunding Announcements (QRAs).

Its anchors are two series from [tdcest](https://github.com/smkwray/tdcest): the quarterly Treasury Deposit Component (TDC), the part of the change in deposits at US banks and credit unions that comes from Treasury payments and trades in Treasury securities, and the weekly Treasury component of reserves (TRC), the change in bank reserves caused by the same Treasury and Federal Reserve transactions. Apart from an offline synthetic demo, every analysis needs local DTS, QRA and `tdcest` files that are not in the repository.

## Data

All external series are public:

- Federal Reserve Economic Data (FRED): H.8 deposits, H.4.1 reserve factors and macro controls listed in `config/series_registry.yml`, from FRED's public `fredgraph.csv` endpoint without an API key.
- Treasury Fiscal Data API: the DTS operating cash balance, deposits and withdrawals of operating cash, income tax refunds, and federal tax deposits.
- From `tdcest`: quarterly TDC tables and hash-pinned H.4.1 and weekly TRC exports, read from `../tdcest` by default (`config/upstream_sources.yml`); commands accept path overrides such as `--source-root`.

## Install

Python 3.11 or later. From the repository root:

```bash
uv venv
uv pip install -e '.[dev]'
```

## Usage

Run the offline synthetic demo (it writes a monthly TDC proxy and validation report to `output/demo/`), then check which upstream files are present:

```bash
uv run --no-sync python -B -m tdchf demo --out output/demo
uv run --no-sync python -B -m tdchf doctor
```

With the inputs in place, run an analysis, for example the weekly TRC projection:

```bash
uv run --no-sync python -B -m tdchf run-weekly-trc-projection
```

Reserve commands read a config under `config/` (here `config/weekly_trc_projection_v1.json`) and stop if an input is missing or fails its SHA-256 check. `python -m tdchf --help` lists every command.

| Commands | What they estimate |
|---|---|
| `build-quarterly-hf-bridge` | Gross DTS credits, drains, debt issuance and redemption by settlement quarter and H.8 bank week (Thursday through Wednesday), reconciled to quarterly `tdcest` series |
| `run-disbursement-response-profile` | Cumulative H.8 deposit projections on gross DTS flows across leads and lags |
| `run-weekly-reserve-offsets`, `run-weekly-trc-projection`, `run-weekly-reserve-history`, `run-weekly-reserve-rolling`, `run-monthly-reserve-rolling` | Reserve offsets on the weekly Treasury cash leg (the fall in the Treasury General Account and Supplementary Financing Program account); core TRC projections, by period with restricted-score wild bootstrap inference; rolling weekly and monthly paths, the monthly ones adding the money-fund route |
| `run-analysis-spec`, `build-qra-borrowing-surprise`, `run-qra-event-lp` | Quarterly TDC allocated to months by YAML workflows in `config/`, and local projections around QRA borrowing surprises |

## Output

CSV tables, Markdown summaries and JSON receipts go to `output/` (for example `output/tables/weekly_trc_projection_v1/`) and, for the DTS and QRA workflows, `data/processed/`. Both directories are ignored by Git.

## Limits

- DTS totals cover all Treasury cash flows and are not matched to bank accounts, so H.8 deposit slopes are unnormalized conditional associations, in billions of deposits per billion dollars of flow.
- Monthly allocations interpolate quarterly anchors; timing within the quarter comes from monthly indicators.
- The H.4.1 reverse-repurchase series for "other counterparties" mixes counterparty types, so it only roughly proxies money-fund settlement through the Federal Reserve's overnight reverse repurchase (ON RRP) facility.

## Tests

The default suite runs offline on fixtures:

```bash
uv run --no-sync python -B -m pytest
uvx ruff check .
```

Tests marked `integration` need local DTS, QRA and state-response data:

```bash
uv run --no-sync python -B -m pytest -m integration
```

## License

MIT. See [LICENSE](LICENSE).
