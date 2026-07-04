#!/usr/bin/env bash
set -euo pipefail

.venv/bin/tdchf run-qra-event-lp \
  --qra data/processed/qra_borrowing_surprise.csv \
  --weekly-panel data/processed/tdc_weekly_channel_panel.csv \
  --out data/processed/qra_event_lp_estimates.csv \
  --readout data/processed/qra_event_lp_readout.md
