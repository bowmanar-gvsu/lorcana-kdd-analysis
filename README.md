# What Wins vs. What Sells
## Knowledge Discovery in Disney Lorcana's Competitive and Collector Markets
**CIS 635 — Knowledge Discovery and Data Mining**
Grand Valley State University | Professor Jiaxin Du | April 2026

---

## Authors
[Artie Bowman](https://github.com/artiebowman), [Mohammad Aziz Boufaied](https://github.com/azizboufaied), [Kennedy Comstock](https://github.com/comstock-ctrl), Nurudeen Showole

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/artiebowman/lorcana-tcg-what-wins-vs-what-sells/blob/main/Bowman_et_al_2026_Lorcana_KDD_Notebook.ipynb) [![Paper](https://img.shields.io/badge/Paper-Open-blue)](https://github.com/artiebowman/lorcana-tcg-what-wins-vs-what-sells/blob/main/paper/Bowman_et_al_2026_What_Wins_vs_What_Sells_Lorcana_KDD.pdf)

## Abstract
This study applies a Knowledge Discovery in Databases (KDD) pipeline to Disney Lorcana, a trading card game with a dual-market ecosystem driven by competitive strategy and Disney IP collector demand. We integrate 2,652 cards from the Lorcast API (2,075 after deduplication), 749 tournament decklists from 710 unique players across 21 events in 13 countries, and three independent price sources spanning May 2025 through March 2026.

We introduce a **dual-axis SHAP framework** comparing feature importance for competitive placement versus market price on a matched feature set — the first such application to a trading card game — and the **Sleeper Card Index (SCI)**, a tournament-based metric bridging competitive performance with market price.

**Central finding:** *What wins and what sells are fundamentally different.* SHAP identifies mechanical density, stat efficiency, and character classifications as competitive drivers, while the market prices rarity and Disney IP recognition. The near-random predictive accuracy of our best model (XGBoost AUC = 0.544) is itself evidence of game balance.

## Key Results

| Metric | Value |
|--------|-------|
| Cards (pre/post dedup) | 2,652 / 2,075 |
| Tournament decks | 749 (710 unique players, 21 events, 13 countries) |
| Full model AUC (87 features) | 0.544 |
| MI-filtered AUC (19 features) | 0.599 |
| Multi-seed mean AUC | 0.547 ± 0.027, 95% CI [0.509, 0.595] |
| Test AUC (March 2026 holdout) | 0.573 |
| TimeSeriesSplit CV AUC | 0.492 (meta drift: Stratified − TimeSeriesSplit = 0.052) |
| Price model R² | 0.44 |
| Rarity SHAP (price model) | 0.207 (5× the #2 feature) |
| Archetype × Top-8 χ² | 11.85, p = 0.540 (not significant) |
| Deck cost vs. Top-8 | r = −0.092, p = 0.012, Cohen's d = −0.230 |
| Spearman placement ρ | +0.132, p < 0.001 (expensive decks place worse) |
| Dogs winner overrepresentation | 1.96× (28.6% of wins from 14.6% of field) |
| SCI validation (temporal) | ρ = 0.291, p < 0.001 |
| Google Trends external validation | ρ = 0.858, p < 0.001 (n = 13 characters) |
| Co-occurrence rules | 1,110 total, 120 cross-archetype (10.8%) |
| WLD records collected | 102 across 7 verified tournaments |
| LLM error audit | 10 errors across 7 events (1.4/event) |

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
├── Bowman_et_al_2026_Lorcana_KDD_Notebook.ipynb  # Full notebook (17 sections, 130 cells)
├── paper/
│   ├── Bowman_et_al_2026_What_Wins_vs_What_Sells_Lorcana_KDD.pdf    # Final paper (PDF)
│   └── Bowman_et_al_2026_What_Wins_vs_What_Sells_Lorcana_KDD.docx   # Final paper (Word)
├── presentation/
│   └── (presentation slides — April 22, 2026)
├── data/
│   ├── lorcana_master_summary.xlsx                 # Combined WLD data (4 sheets)
│   └── README_data.md                              # Data source documentation
└── figures/
    ├── fig_01_meta_share.png
    ├── fig_02_pca_clusters.png
    ├── fig_03_shap_beeswarm.png
    ├── fig_04_price_shap.png
    ├── fig_05_dual_axis_bars.png
    ├── fig_06_dual_axis_scatter.png
    ├── fig_07_deck_cost.png
    ├── fig_08_network.png
    ├── fig_09_sci_bars.png
    ├── fig_10_ablation.png
    └── ...                                          # All figures at 300 DPI
```

## Methods

### Data Sources
- **Lorcast API** — 2,652 cards with mechanics, stats, classifications, daily listing prices (2,262 cards)
- **inkdecks.com** — 749 tournament decklists hand-collected from 21 official DLC/CCQ events
- **tcgcsv.com** — Market prices for 2,906 entries (snapshot: April 10, 2026)
- **Periodic price snapshots** — 17 monthly windows (May 2025–March 2026) for temporal reconstruction
- **lorcana_master_summary.xlsx** — WLD records, tournament metadata, and LLM error audit from multi-source aggregation

### Pipeline (KDD Lifecycle)

| Step | Notebook Sections | Techniques |
|------|-------------------|------------|
| Selection | §2 | Lorcast API, inkdecks.com scraping, tcgcsv.com prices |
| Preprocessing | §3–4 | 33 mechanic flags, 35 classifications, dual-ink patching, fuzzy matching (448/449), deduplication |
| Transformation | §5–6 | 13 engineered features, TF-IDF (10 SVD components), deck-level aggregation, 87 total features |
| Mining (Predictive) | §8–9 | XGBoost + RF + LR, MI feature selection (19 MI-filtered for SHAP), dual CV (Stratified + Temporal), dual-axis SHAP, price regression |
| Mining (Descriptive) | §7, 10–11 | K-Means clustering (ARI = 0.807), co-occurrence network (Louvain), FP-Growth association rules |
| Evaluation | §13 | Ablation (5 layers), temporal holdout, chi-squared, Cohen's d, multi-seed stability, placement regression |
| Interpretation | §12, 14 | Sleeper Card Index, temporal SCI validation, external validation (Google Trends ρ = 0.858) |

### Novel Contributions
1. **Dual-axis SHAP** — Side-by-side comparison of competitive vs. price feature importance on a matched feature set
2. **Sleeper Card Index (SCI)** — Tournament-based metric: competitive percentile rank − price percentile rank
3. **Near-random AUC as balance evidence** — Low predictive accuracy demonstrates healthy game design where player skill dominates over deck composition
4. **Custom dual-ink patching** — Recovered ink assignments for 102/121 dual-ink cards missing from API

## Running the Notebook
1. Open `Bowman_et_al_2026_Lorcana_KDD_Notebook.ipynb` in **Google Colab**
2. Run §1 (Setup & Imports) to install dependencies
3. Cells run sequentially — each section builds on previous outputs
4. §15 (Paper Number Verification) validates all numbers cited in the paper
5. §16 exports all figures at 300 DPI to Google Drive

### Dependencies
```
pandas, numpy, scikit-learn, xgboost, shap, networkx, python-louvain,
mlxtend, matplotlib, seaborn, fuzzywuzzy, python-Levenshtein
```

## Not Pay-to-Win
The cheapest competitive archetype (Dogs, $179 median) has the highest Top-8 rate (24.8%) and is overrepresented among tournament winners (1.96× ratio, median placement 18th vs 32nd for the most-played archetype). The most expensive archetype (Blurple, $459) wins less often (18.8%). Deck cost is negatively correlated with winning (r = −0.092, p = 0.012). Spearman rank correlation on exact placement (1st–128th) confirms this across the full range (ρ = +0.132, p < 0.001). Tournament winners have a median deck cost of $277 versus $316 overall.

## Repository Status
This repository accompanies the April 2026 class submission. A peer-reviewed version targeting IEEE CoG 2026 is in preparation; notebook updates and methodology refinements will land between May and August 2026.

## Acknowledgments
AI tools (Claude, Anthropic) were used as a coding assistant for data pipeline development, visualization formatting, iterative paper editing and auditing, and document preparation. All analytical decisions, statistical methodology, domain interpretations, and written findings are the authors' own work.

The authors thank Prof. Jiaxin Du for valuable feedback and for encouraging publication of this work.

## License
This project began as academic research and is being developed for publication and community use. Data sourced from Lorcast API, inkdecks.com, and tcgcsv.com under their respective terms of service. Disney Lorcana is a trademark of Disney/Ravensburger.

## Citation
If you use this work, please cite:
```
Bowman, A., Boufaied, M.A., Comstock, K., & Showole, N. (2026).
What Wins vs. What Sells: Knowledge Discovery in Disney Lorcana's
Competitive and Collector Markets. CIS 635, Grand Valley State University.
```
