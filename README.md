# What Wins vs. What Sells

## Knowledge Discovery in Disney Lorcana's Competitive and Collector Markets

**CIS 635 — Knowledge Discovery and Data Mining**
Grand Valley State University | Professor Jiaxin Du | April 2026

---

## Authors

Artie Bowman, Mohammad Aziz Boufaied, Kennedy Comstock, Nurudeen Showole

## Abstract

This study applies a Knowledge Discovery in Databases (KDD) pipeline to Disney Lorcana, a trading card game with a dual-market ecosystem driven by competitive strategy and Disney IP collector demand. We integrate 2,652 cards from the Lorcast API (2,075 after deduplication), 749 tournament decklists from 710 unique players across 21 events in 13 countries, and three independent price sources spanning May 2025 through March 2026.

We introduce a **dual-axis SHAP framework** comparing feature importance for competitive placement versus market price — to our knowledge, a first application of this technique in any domain — and the **Sleeper Card Index (SCI)**, a tournament-based metric bridging competitive performance with market price.

**Central finding:** *What wins and what sells are fundamentally different.* SHAP identifies mechanical density, stat efficiency, and character classifications as competitive drivers, while the market prices rarity and Disney IP recognition. The near-random predictive accuracy of our best model (XGBoost AUC = 0.544) is itself evidence of game balance.

## Key Results

| Metric | Value |
|--------|-------|
| Cards (pre/post dedup) | 2,652 / 2,075 |
| Tournament decks | 749 (710 unique players, 21 events, 13 countries) |
| Full model AUC (86 features) | 0.544 |
| MI-filtered AUC (19 features) | 0.599 |
| Multi-seed mean AUC | 0.547 ± 0.027, 95% CI [0.509, 0.595] |
| Test AUC (March 2026 holdout) | 0.573 |
| Price model R² | 0.44 |
| Rarity SHAP (price model) | 0.207 (5× the #2 feature) |
| Archetype × Top-8 χ² | 11.85, p = 0.540 (not significant) |
| Deck cost vs. Top-8 | r = −0.092, p = 0.012, Cohen's d = −0.230 |
| SCI validation (temporal) | ρ = 0.291, p < 0.001 |
| Co-occurrence rules | 1,110 total, 120 cross-archetype (10.8%) |

## SHAP: What Wins vs. What Sells

| Rank | Competitive (What Wins) | Price (What Sells) |
|------|------------------------|--------------------|
| #1 | tfidf_comp_0 (mechanical density) | Rarity (0.207 mean SHAP) |
| #2 | avg_vanilla_score (stat efficiency) | Ink cost |
| #3 | avg_keyword_count (card complexity) | Set number |
| #4 | pct_cls_mentor (Mentor classification) | On-play effects |
| #5 | pct_evasive (Evasive density) | Card type |

**Rarity** is the #1 predictor of market price but was **not even selected** by the MI filter for competitive modeling (MI < 0.01). This is the central divergence.

## Repository Structure

```
├── README.md
├── Lorcana_KDD_WhatWins_vs_WhatSells_V3.ipynb   # Full notebook (17 sections, 130 cells)
├── paper/
│   └── Lorcana_KDD_Paper_V9_Final.docx           # Final paper (dual-column, ~9,000 words)
├── presentation/
│   └── Lorcana_KDD_Presentation.pptx              # 16-slide presentation
├── data/
│   ├── lorcana_master_summary.xlsx                 # Combined WLD data
│   └── README_data.md                              # Data source documentation
└── figures/
    ├── fig_01_stat_distributions.png
    ├── fig_04_meta_share.png
    ├── fig_05_pca_clusters.png
    ├── fig_07_shap_xgb.png                         # SHAP beeswarm (19 MI-filtered features)
    ├── fig_09_price_shap.png
    ├── fig_11_dual_axis_bars.png                    # Dual-axis SHAP comparison
    ├── fig_12_dual_axis_scatter.png                 # Feature rank divergence scatter
    ├── fig_13_deck_cost.png
    ├── fig_15_network.png                           # Co-occurrence network (Louvain communities)
    ├── fig_17_sci_bars.png                          # Sleeper Card Index
    ├── fig_19_ablation.png                          # Feature engineering ablation
    └── ...                                          # 21 total figures at 300 DPI
```

## Methods

### Data Sources
- **Lorcast API** — 2,652 cards with mechanics, stats, classifications, daily listing prices (2,262 cards)
- **inkdecks.com** — 749 tournament decklists hand-collected from 21 official DLC/CCQ events
- **tcgcsv.com** — Market prices for 2,906 entries (snapshot: April 10, 2026)
- **Periodic price snapshots** — 17 monthly windows (May 2025–March 2026) for temporal reconstruction

### Pipeline (KDD Lifecycle)

| Step | Sections | Techniques |
|------|----------|------------|
| Selection | §2 | Lorcast API, inkdecks.com scraping, tcgcsv.com prices |
| Preprocessing | §3–4 | 33 mechanic flags, 35 classifications, dual-ink patching, fuzzy matching (448/449), deduplication |
| Transformation | §5–6 | 13 engineered features, TF-IDF (10 SVD components), deck-level aggregation, 86 total features |
| Mining (Predictive) | §8–9 | XGBoost + RF + LR, MI feature selection, dual CV (Stratified + Temporal), dual-axis SHAP, price regression |
| Mining (Descriptive) | §7, 10–11 | K-Means clustering (ARI = 0.807), co-occurrence network (Louvain), FP-Growth association rules |
| Evaluation | §13 | Ablation (5 layers), temporal holdout, chi-squared, Cohen's d, multi-seed stability, placement regression |
| Interpretation | §12, 14 | Sleeper Card Index, temporal SCI validation, external validation (Google Trends ρ = 0.831) |

### Novel Contributions
1. **Dual-axis SHAP** — Side-by-side comparison of competitive vs. price feature importance
2. **Sleeper Card Index (SCI)** — Tournament-based metric: competitive percentile rank − price percentile rank
3. **Near-random AUC as balance evidence** — Low predictive accuracy reframed as game design validation
4. **Custom dual-ink patching** — Recovered ink assignments for 102/121 dual-ink cards missing from API

## Running the Notebook

1. Open `Lorcana_KDD_WhatWins_vs_WhatSells_V3.ipynb` in **Google Colab**
2. Run §1 (Setup & Imports) to install dependencies
3. Cells run sequentially — each section builds on previous outputs
4. §15 (Paper Number Verification) validates all numbers cited in the paper
5. §16 exports all 21 figures at 300 DPI to Google Drive

### Dependencies
```
pandas, numpy, scikit-learn, xgboost, shap, networkx, python-louvain,
mlxtend, matplotlib, seaborn, fuzzywuzzy, python-Levenshtein
```

## Not Pay-to-Win

The cheapest competitive archetype (Dogs, $179 median) has the highest Top-8 rate (24.8%) and is overrepresented among tournament winners (1.96× ratio). The most expensive archetype (Blurple, $459) wins less often (18.8%). Deck cost is negatively correlated with winning (r = −0.092, p = 0.012).

## Acknowledgments

AI tools (Claude, Anthropic) were used as a coding assistant for data pipeline development, visualization formatting, iterative paper editing and auditing, and document preparation. All analytical decisions, statistical methodology, domain interpretations, and written findings are the authors' own work.

The authors thank Prof. Jiaxin Du for valuable feedback and for encouraging publication of this work.

## License

This project is for academic purposes. Data sourced from Lorcast API, inkdecks.com, and tcgcsv.com under their respective terms of service. Disney Lorcana is a trademark of Disney/Ravensburger.

## Citation

If you use this work, please cite:
```
Bowman, A., Boufaied, M.A., Comstock, K., & Showole, N. (2026).
What Wins vs. What Sells: Knowledge Discovery in Disney Lorcana's
Competitive and Collector Markets. CIS 635, Grand Valley State University.
```
