# FairAudit-KZ — prototype (TRL 4)

Audit → explain → mitigate pipeline for group bias in credit-style ML models.
Research Tools and Methods, Assignment 12, SDU University. Author: Turlan, ID 230103373.

## Reproduce
```bash
pip install scikit-learn pandas matplotlib fairlearn shap
python fairaudit.py --data adult.data --out results   # ~70 s on a laptop CPU
python gantt.py
```
Data: UCI Adult (Becker & Kohavi, 1996), 48,842 rows → 45,222 complete records. Protected attribute: sex.

## Outputs (`results/`)
- `summary.csv` — mean ± sd over 5 seeds for 8 configurations
- `raw_results.csv` — every seed
- `audit.json` — proxy-detector AUC, SHAP top-10, success-criteria check
- `fig_tradeoff.png`, `fig_shap.png`, `fig_gantt.png`

## Pre-registered success criteria (vs GBM baseline)
DP gap reduced ≥ 50 %, accuracy loss ≤ 2 pp, mean disparate-impact ratio ≥ 0.8.
Result: FairAudit (reweighing + validation-tuned threshold optimiser) — DP −80 %, −1.83 pp, DI = 0.825 → **met**.

Licence: Apache 2.0 (code). Dataset: CC BY 4.0 (UCI).
