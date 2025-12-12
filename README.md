# GRAFT: Grid-Aware Load Forecasting with Multi-Source Textual Alignment and Fusion

This repository provides a minimal, research-oriented implementation accompanying the paper:

**GRAFT: Grid-Aware Load Forecasting with Multi-Source Textual Alignment and Fusion**

The codebase supports grid-aware electric load forecasting and optional integration of aligned external textual signals (News / Reddit / Policy) into rolling forecasts, with interpretable fusion mechanisms.

---

## Contents (minimal release)

- `run.py`: interactive training entry (state selection + external source selection).
- `exp_stanhop_fiats.py`: main experiment script (non-interactive / research usage).
- `code_models/`: model definitions (STanHop/FIATS/fusion modules).
- `utils/`: data utilities, metrics, and helper functions.

> This repository intentionally keeps a minimal structure for clarity.  
> Raw datasets and large outputs are not tracked by Git.

---

## Environment

### Option A: pip (recommended)
```bash
pip install -r requirements.txt
