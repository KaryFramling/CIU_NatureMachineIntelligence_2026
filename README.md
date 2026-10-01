# CIU_NatureMachineIntelligence_2026

Code accompanying the paper:

> Främling, K. (2026). **Re-founding Explainable AI: Contextual Importance and Utility
> as a Universal Explanation Framework.**
> *Nature Machine Intelligence* (submitted).

This repository reproduces all programmatically generated figures in the paper.

---

## Repository structure

```
CIU_NatureMachineIntelligence_2026/
│
├── wealth_example/
│   └── ciu_vs_shap_wealth.py   # Figs 2–3: IO plots, PI plot, SHAP comparison
│
├── empirical/
│   └── NMIFigures.R            # Figs 4–9: German Credit & Ames Housing results
│
├── requirements.txt            # Python dependencies
├── R_requirements.txt          # R package dependencies
└── README.md
```

---

## Wealth example (Python) — Figures 2–3

Generates IO plots, a CIU Potential Influence (PI) plot, and a SHAP comparison
for the linear wealth toy model used throughout the paper.

**Requirements:** Python ≥ 3.9, numpy, matplotlib

```bash
pip install -r requirements.txt
cd wealth_example
python ciu_vs_shap_wealth.py
```

Output PDFs and PNGs are written to `wealth_example/figures/`.

---

## Empirical demonstrations (R) — Figures 4–9

Trains Random Forest (German Credit) and GBM (Ames Housing) models via `caret`,
computes CIU values using the `ciu` R package, and generates all empirical figures.

**Requirements:** R ≥ 4.3, packages listed in `R_requirements.txt`

```r
source("empirical/NMIFigures.R")
```

Output PDFs are written to `empirical/figures/`.

---

## CIU R package

The `ciu` package used in the empirical section is available at:
<https://github.com/KaryFramling/ciu>

Install from CRAN:
```r
install.packages("ciu")
```

Or the development version from GitHub:
```r
# install.packages("remotes")
remotes::install_github("KaryFramling/ciu")
```

---

## Citation

```bibtex
@article{Framling2026NMI,
  author  = {Främling, Kary},
  title   = {Re-founding Explainable {AI}: {C}ontextual Importance and Utility
             as a Universal Explanation Framework},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Licence

MIT — see `LICENSE`.
