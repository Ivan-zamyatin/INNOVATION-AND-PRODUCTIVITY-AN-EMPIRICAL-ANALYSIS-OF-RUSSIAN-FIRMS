# R&D and Total Factor Productivity in Russia (2011–2023)

> Bachelor's thesis — empirical estimation of the dynamic, state-dependent return to R&D investment on a panel of Russian production-economy firms, with a custom Levinsohn–Petrin specification in which R&D capital is identified as a state variable, and a full Doraszelski–Jaumandreu (2013) endogenous-Markov productivity transition.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-pdflatex-008080.svg)](https://www.latex-project.org/)
[![Data: RFSD](https://img.shields.io/badge/data-RFSD%20HuggingFace-yellow.svg)](https://huggingface.co/datasets/irlspbru/RFSD)
[![Models: 6 estimators](https://img.shields.io/badge/estimators-OLS%20·%20FE%20·%20FE+time%20·%20OP%20·%20LP%20·%20DJ-orange.svg)]()

---

- **Question.** Does R&D investment causally raise firm-level Total Factor Productivity (TFP) in the Russian production economy, and is the return uniform across firms?
- **Data.** A panel of **691,644 firm-year observations** on **162,887 unique firms** from the [Russian Financial Statements Database (RFSD)](https://huggingface.co/datasets/irlspbru/RFSD), 2011–2023, restricted to the eight production-oriented OKVED sections (A, B, C, D, E, F, H, J).
- **Methods.** A hierarchy of six estimators — pooled OLS, entity FE, two-way FE, Olley–Pakes (1996), Levinsohn–Petrin (2003) with the Ackerberg–Caves–Frazer (2015) correction, and Doraszelski–Jaumandreu (2013) endogenous productivity transition with a complete degree-3 polynomial — estimated on the pooled panel and replicated sector-by-sector. Inference: clustered panel bootstrap, *B = 200* replications.
- **Methodological contribution.** A custom LP–ACF specification in which **R&D capital is treated symmetrically with physical capital as a state variable**, allowing $\beta_r$ to be identified jointly with $(\beta_l, \beta_m, \beta_k)$ in the second-stage GMM. Hansen $J$-test does not reject the over-identifying restrictions ($p = 0.114$).
- **Headline result.** Structural identification yields $\beta_r^{\text{LP}} = 0.0150$ ($p < 0.01$) and $\beta_r^{\text{OP}} = 0.0123$ ($p < 0.05$). The DJ transition function reveals **state-dependent returns**: marginal effect of lagged R&D decays monotonically from $+0.008$ at the bottom decile of within-sample $\omega$ to $+0.001$ at the top decile, while remaining positive throughout — empirical realisation of the *absorptive-capacity* prediction (Audretsch 2020) and the *Aghion–Howitt step-by-step competition* prior.

---

## Cross-sectoral highlights

| Insight | Mechanism |
|---|---|
| **IT paradigm shift** | In information & communication, $\beta_l$ rises to 0.27 (LP) / 0.31 (DJ) and $\beta_m$ falls to 0.66 — human capital displaces intermediate inputs as the marginal driver of value-added. |
| **Infrastructure beats high-tech on R&D** | Highest $\beta_r$ in transportation (0.041) and water supply (0.022); manufacturing (0.003) and IT (0.005) sit at the bottom. Mature sectors are in the diminishing-returns zone; sluggish infrastructure has wide catch-up margin. |
| **The "ghost of capital"** | Across all 8 sectors, $\beta_k \in [0.001, 0.03]$ under LP/DJ — within-firm output is essentially regulated by materials and labour, not by capital. Suggests under-utilisation, deep depreciation, or inelastic investment budgets. |
| **Logistics DRS** | Transportation: RTS = 0.825 in both LP and DJ — strongest decreasing returns in the panel. Combined with the high $\beta_r$, the picture is one of binding capacity constraints that R&D could relax. |

---

## Repository structure

```
.
├── README.md                                   # this file
├── code.ipynb                    # main analysis notebook (53 cells)
└── thesis
```

---

## Reproducibility

### Prerequisites

| Component | Version | Purpose |
|---|---|---|
| Python | 3.11 | analysis pipeline |
| pandas, numpy, scipy | latest | data wrangling, optimisation |
| polars | latest | fast parquet I/O for raw RFSD shards |
| statsmodels | latest | OLS, OP probit survival stage |
| linearmodels | latest | PanelOLS for FE / FE+time |
| seaborn, matplotlib | latest | figure generation |
| sklearn | latest | `PolynomialFeatures` for proxy polynomials |
| joblib, tqdm | latest | parallel bootstrap |
| Jupyter | any | notebook execution |
| TeX Live (pdflatex + biber) | 2023+ | manuscript compilation |

### Running the analysis

1. **Clone the repository and prepare data.** The notebook will download the raw RFSD parquet shards from HuggingFace on the first run if `rfsd_*.parquet` files are not yet present in the project root. The cleaned, sector-filtered panel `rfsd_final_panel.csv` is regenerated automatically from the parquet shards.

2. **Execute the notebook end to end.** The full pipeline runs in approximately **3 hours** with the production setting `N_BOOT = 200` (cell 52, *Main Execution Block*), and in approximately **15 minutes** with the development setting `N_BOOT = 20`.

   ```bash
   jupyter nbconvert --to notebook \
       --execute code.ipynb \
       --output code.ipynb \
       --ExecutePreprocessor.timeout=14400
   ```

### What the notebook produces

| Artifact | Source cell | Destination |
|---|---|---|
| Cleaned panel CSV | cells 7–24 | `rfsd_final_panel.csv` |
| Pooled estimates with bootstrap SEs | cells 32, 34, 44 | `results/production_functions_results.csv` |
| All 18 LaTeX tables | cell 50 | `results/all_tables.tex` |
| All figures (300 dpi PNG) | cells 14, 42, 46 | `results/plots/`, `eda_export_<timestamp>/` |

---

## Data

**Source.** [Russian Financial Statements Database (RFSD)](https://huggingface.co/datasets/irlspbru/RFSD), 2011–2023 yearly shards (Bondarkov et al. 2025). Extracted columns:

| RFSD line / column | Variable in code | Economic meaning |
|---|---|---|
| `inn` | `inn` | Firm tax identifier |
| `okved_section` | `sector` | OKVED-2 section letter (A–U) |
| `oktmo` | `oktmo` | Regional code |
| `year` | `year` | Reporting year |
| `line_2110` | `PL_revenue` | Revenue (output) |
| `line_1150` | `B_fixed_assets` | Fixed assets (capital, balance-sheet) |
| `line_2120` | `PL_cost_of_sales` | Cost of sales (intermediate inputs) |
| `line_1120` | `B_research_development` | R&D capital (balance-sheet) |
| `line_4122` | `CFo_labor` | Labour cash outflows |

**Sectoral filter.** Eight OKVED sections retained for the Cobb–Douglas framework: A (agriculture), B (mining), C (manufacturing), D (electricity & gas), E (water supply & sanitation), F (construction), H (transportation & storage), J (information & communication).

**Cleaning.** A multi-criteria smart filter removes economically anomalous observations (zero or negative production inputs, dormant entities, dead holdings, active shells, ghost firms), followed by symmetric winsorisation at the 0.5%/1% tails on the four positive inputs.

**Deflation.** Flow variables (revenue, cost of sales, labour outflows) are deflated to constant 2011 prices using monthly-CPI-chain-derived annual averages; balance-sheet stocks (fixed assets, R&D) are deflated using end-of-December cumulative price levels.

**R&D capital stock.** Constructed via the perpetual inventory method (PIM) with $\delta = 0.15$:

$$S_t = (1 - \delta) \cdot S_{t-1} + R_t$$

The regression input throughout the manuscript is $r_{it} = \ln(S_{it} + 1)$.

---

## Methodology

### Estimator hierarchy

The paper estimates a hierarchy of six specifications. Each layer isolates a distinct source of identification failure:

| Estimator | What it absorbs | What it does not |
|---|---|---|
| Pooled OLS | — | Selection, simultaneity, endogenous transition |
| Entity FE | Time-invariant firm heterogeneity | Simultaneity, endogenous transition |
| FE + time | + common time shocks | Simultaneity, endogenous transition |
| Olley–Pakes (1996) | + simultaneity (via investment proxy) | Endogenous transition |
| LP–ACF (2003 / 2015), R&D as state | + simultaneity (via materials proxy) + identifies $\beta_r$ | Endogenous transition |
| Doraszelski–Jaumandreu (2013) | + endogenous Markov transition $g(\omega_{t-1}, r_{t-1})$ | — (preferred specification) |

### LP–ACF with R&D as state variable (custom contribution)

In the standard LP–ACF setup R&D enters only as a polynomial argument in the first-stage proxy and is therefore inseparable from $\omega_{it}$ in the second-stage GMM, so $\beta_r$ remains unidentified within LP. The thesis modifies the specification by treating $r_{it}$ symmetrically with physical capital as a **state variable**:

```python
LevinsohnPetrinModel(
    df, y_col='l_y',
    free_cols=['l_l'],
    state_cols=['l_k', 'l_r'],   # ← R&D added as state variable
    proxy_col='l_m',
    poly_degree=3,
)
```

The Stage-2 moment conditions

$$\mathbb{E}[\xi_{it}(\beta) \cdot Z_{it}] = 0, \quad
Z_{it} = (l_{it-1}, m_{it-1}, k_{it-1}, r_{it-1}, k_{it}, r_{it})$$

deliver six instruments for four parameters $(\beta_l, \beta_m, \beta_k, \beta_r)$. The Hansen $J$-test does not reject the over-identifying restrictions ($J = 4.35$, $p = 0.114$).

### Doraszelski–Jaumandreu (2013) transition function

The productivity Markov law

$$\omega_{it} = g(\omega_{it-1}, r_{it-1}) + \xi_{it}$$

is approximated with a complete third-degree polynomial in $(\omega_{it-1}, r_{it-1})$:

$$
g(\omega, r) = \gamma_0
+ \gamma_1 \omega + \gamma_2 r
+ \gamma_3 \omega^2 + \gamma_4 r^2 + \gamma_5 \omega r
+ \gamma_6 \omega^3 + \gamma_7 r^3 + \gamma_8 \omega^2 r + \gamma_9 \omega r^2.
$$

The cross-derivative $\partial^2 g / \partial \omega \partial r = \gamma_5 + 2\gamma_8 \omega + 2\gamma_9 r$ is allowed to vary across firms. A degree-2 sensitivity check is reported in Appendix~D.

### Inference

- **OLS** uses MacKinnon–White HC3 heteroscedasticity-consistent standard errors.
- **FE / FE+time** use one-way cluster-robust covariance at the firm (INN) level.
- **OP / LP / DJ** use a clustered panel bootstrap with $B = 200$ firm-level resampling replications.

---

## Results summary

### Pooled-panel estimates

| | OLS | FE | FE+time | OP | LP | DJ |
|---|---|---|---|---|---|---|
| $\beta_l$ (labour) | 0.0892\*\*\* | 0.1365\*\*\* | 0.1376\*\*\* | 0.0690\*\*\* | 0.1307\*\*\* | 0.1306\*\*\* |
| $\beta_m$ (materials) | 0.8805\*\*\* | 0.8030\*\*\* | 0.8025\*\*\* | 0.8827\*\*\* | 0.8414\*\*\* | 0.8420\*\*\* |
| $\beta_k$ (capital) | 0.0044\*\*\* | 0.0260\*\*\* | 0.0251\*\*\* | 0.0493\*\*\* | 0.0010\*\*\* | 0.0010\*\*\* |
| $\beta_r$ (R&D capital) | 0.0209\*\*\* | 0.0037\* | 0.0046\*\* | 0.0123\*\* | 0.0150\*\*\* | — |
| Returns to scale | 0.974 | 0.966 | 0.965 | 1.001 | 0.973 | 0.974 |
| Persistence ($\rho$) | 0.7193 | 0.1167 | 0.1161 | 0.9872 | 0.7302 | 0.7303 |
| $R^2$ | 0.9359 | 0.8349 | 0.8302 | — | — | — |
| Hansen $J$ ($p$-value) | — | — | — | — | 4.35 ($p$=0.114) | 8.14 ($p$=0.017) |

\* $p < 0.1$; \*\* $p < 0.05$; \*\*\* $p < 0.01$. Clustered panel-bootstrap SEs ($B = 200$). Full table: `results/all_tables.tex` → `tab:main_results`.

### DJ marginal effect of lagged R&D, by within-sample $\omega$ quantile

| $\omega$ quantile | $\partial \mathbb{E}[\omega_t] / \partial r_{t-1}$ |
|---|---|
| q10 ($\omega = +0.331$) | +0.0084 |
| q25 ($\omega = +0.551$) | +0.0083 |
| q50 ($\omega = +0.752$) | +0.0071 |
| q75 ($\omega = +0.940$) | +0.0051 |
| q90 ($\omega = +1.173$) | +0.0014 |

Positive, decaying monotonically with productivity. The cross-derivative $\partial^2 g/\partial\omega\partial r$ is uniformly negative on the in-sample support (100% of observations); its absolute magnitude grows from $|-0.007|$ in the bottom decile to $|-0.021|$ in the top decile. Both findings are consistent with the absorptive-capacity prediction.

---

## Manuscript

Section roadmap:

1. Declaration of the Use of Generative Models (HSE-mandated AI-use disclosure).
2. Abstract.
3. Introduction.
4. Literature Review.
5. Methods (Models + Estimation Strategy with Identification Logic).
6. Data and Variables (Data Source + Variables Construction + Descriptive Analysis).
7. Results (Production Function Estimates + Dynamic Returns to R&D + Cross-sectoral Analysis).
8. Discussion (Headline + State-dependent returns + Comparison with prior empirical evidence + R&D premium dynamics + Refining theoretical lenses + Implications + Limitations).
9. Conclusion.
10. Appendices: A. Data and Code Availability; B. Anomaly Identification and Filtering Protocol; C. Supplementary Model Results and Diagnostics; D. DJ Polynomial-Order Sensitivity; E. Cross-Sectoral Estimation by Method.

---

## Citation

If you build on this work, please cite as:

```bibtex
@thesis{zamatin2026rd,
  author       = {Zamatin, Ivan},
  title        = {INNOVATION AND PRODUCTIVITY: AN EMPIRICAL ANALYSIS OF RUSSIAN FIRMS},
  type         = {Bachelor's thesis},
  school       = {HSE University},
  year         = {2026}
}
```

The underlying data are due to:

```bibtex
@article{bondarkov2025russian,
  author  = {Bondarkov, S. and others},
  title   = {The Russian Financial Statements Database (RFSD)},
  journal = {Nature Scientific Data},
  year    = {2025}
}
```

---

## Acknowledgements

- Data: Russian Financial Statements Database (RFSD), Bondarkov et al. (2025), distributed via [HuggingFace Datasets](https://huggingface.co/datasets/irlspbru/RFSD).
- Methodological foundations: Olley & Pakes (1996), Levinsohn & Petrin (2003), Ackerberg, Caves & Frazer (2015), Doraszelski & Jaumandreu (2013), Griliches (1979), Aghion & Howitt (1990), Audretsch (2020).
- Generative-model use disclosure: see Section 1 of the manuscript (`thesis.pdf`), which lists per-section the use of Anthropic Claude (code refactoring, manuscript editing) and Google Gemini (translation, literature triage), in compliance with Section 2 of the HSE *Regulations on Checking Student Papers for Plagiarism, the Use of Generative Models, and the Publication of Bachelor's, Specialist, and Master's Theses on the HSE University Corporate Website*.

---

## License

The underlying RFSD data are subject to their original licence as distributed by the dataset authors on HuggingFace; please consult the dataset card before redistributing.
