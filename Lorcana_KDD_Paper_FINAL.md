---
title: |
  What Wins vs. What Sells: Knowledge Discovery in Disney Lorcana's Competitive and Collector Markets
subtitle: "A KDD Approach to Deck Optimization, Game Balance, and Market Divergence"
author: "Artie Bowman, Mohammad Aziz Boufaied, Kennedy Comstock, Nurudeen Showole"
date: "April 2026"
institute: |
  CIS 635 -- Knowledge Discovery and Data Mining\
  Professor Jiaxin Du\
  Grand Valley State University
abstract: |
  This study applies a Knowledge Discovery in Databases (KDD) pipeline to Disney Lorcana, a trading card game with a dual-market ecosystem driven by competitive strategy and Disney IP collector demand. Drawing on 2,065 unique cards, 753 post-rotation decklists across 21 tournaments (September 2025--March 2026), and 9 price snapshots covering 407 cards, we construct an integrated framework spanning TF-IDF text extraction, K-Means clustering, Random Forest and XGBoost classification with SHAP explainability, NetworkX graph analysis, FP-Growth association rules, Prophet forecasting, and a novel Sleeper Card Index (SCI). Our central finding: *what wins and what sells are different.* SHAP identifies deck-level optimization features as Top-8 predictors, while keyword traits like Evasive are universal "table stakes." Models fail to exceed majority-class baselines on held-out data, and no model across seven configurations achieves both competitive accuracy and meaningful Top-8 recall. Placement percentile regression reveals that deck composition provides a statistically significant but weak directional signal ($\rho = 0.235$, R$^2$ = 0.007), with over 99% of placement variance unexplained by deck composition---evidence of healthy game balance. Price regression yields negative R-squared, confirming traits predict competitive presence but not market value. The SCI reveals systematic underpricing of competitive staples alongside collector premiums on Disney IP cards. Multi-modal ablation shows +85.7% RMSE improvement when combining numeric, text, and network features. A plain-language summary of competitive guidelines for players is provided in Section 11.5.
keywords: "KDD, collectible card games, Disney Lorcana, SHAP, game balance, market divergence, association rules, network analysis"
geometry: margin=0.75in
fontsize: 10pt
documentclass: article
classoption: twocolumn
header-includes:
  - \usepackage{times}
  - \usepackage{graphicx}
  - \usepackage{booktabs}
  - \usepackage{amsmath}
  - \usepackage{float}
  - \usepackage{caption}
  - \captionsetup{font=small}
  - \setlength{\columnsep}{0.3in}
---

# 1. Introduction

## 1.1 Background and Motivation

Collectible card games (CCGs) occupy a unique intersection of combinatorial strategy, game theory, and speculative finance. As Hoover et al. (2020) observe, CCGs are "relatively understudied in the AI community" despite posing challenges that arguably exceed those of classic board games: unlike Go or Chess, CCGs combine imperfect information (hidden hands and deck order), stochastic elements (random draws), and a two-phase optimization problem (deck construction followed by in-game play)---each dimension adding complexity absent from perfect-information games. Since the 1993 launch of Magic: The Gathering, CCGs have generated multi-billion-dollar secondary markets in which individual cards command prices ranging from fractions of a cent to several thousand dollars. Disney Lorcana, released by Ravensburger in August 2023, entered this space with strong commercial momentum, drawing on globally recognized Disney intellectual property.

Disney Lorcana is a CCG where players use a 60-card deck containing characters, items, and actions to become the first player to accumulate 20 lore---accomplished through strategic interactions with active cards. With over 2,000 cards currently available across 11 released sets, and professional tournaments stationed worldwide with substantial cash prizes, both competitive players and collectors benefit from understanding which card traits drive success.

Lorcana is distinguished from prior-studied CCGs by its *dual-market structure*: competitive utility and Disney IP collector appeal independently drive prices. A card featuring a beloved character (e.g., Elsa, Baymax) may command a premium even without competitive relevance, while a strategically dominant but visually unremarkable card may be underpriced relative to its tournament impact. This dynamic creates systematic market inefficiencies that a well-constructed KDD pipeline can surface.

## 1.2 Research Questions

This work is motivated by three primary research questions:

**RQ1 (What Wins?):** Which intrinsic card attributes, engineered efficiency metrics, and latent textual features most strongly predict competitive Top-8 tournament placement?

**RQ2 (What Sells?):** Do the same traits that predict competitive success also predict secondary market price, or do these markets diverge?

**RQ3 (Game Balance):** Can deck composition alone predict tournament outcomes, or does the game exhibit evidence of healthy competitive balance where player skill dominates?

## 1.3 Novelty Statement

While prior work has applied data mining to Magic: The Gathering (Pawlicki et al., 2014; Fink et al., 2015) and Hearthstone (Mora et al., 2022), and predictive sports analytics have demonstrated the broader applicability of classification-based outcome prediction (Leung & Joseph, 2014), Disney Lorcana has not been formally studied in the academic KDD literature. Our contributions include: (1) the first end-to-end KDD pipeline for Disney Lorcana; (2) a novel Sleeper Card Index (SCI) that surfaces cards whose competitive value exceeds their market price; (3) a three-layer framework for competitive advantage distinguishing "table stakes" traits from optimization axes and card packages; (4) empirical evidence for game balance through four independent analytical methods; and (5) quantification of the collector-competitor market divergence.

## 1.4 Professional Significance

Beyond its academic contribution, this work demonstrates the full KDD lifecycle applied to a real-world, multi-modal domain. The methods employed---data fusion, NLP-based feature engineering, clustering, explainable supervised learning, graph analytics, frequent pattern mining, and time-series forecasting---map directly to core competencies in data science practice. The domain's blend of structured numerical data, unstructured text, network-level interactions, and temporal market behavior makes it a strong demonstration for modern knowledge discovery methods.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 describes our methodology and the KDD process. Section 4 details data collection and preprocessing. Section 5 presents feature engineering. Sections 6--8 report results on competitive prediction, market analysis, and network synergy. Section 9 introduces the Sleeper Card Index. Section 10 presents ablation studies. Section 11 discusses findings, implications, and limitations. Section 12 concludes.

# 2. Related Work

## 2.1 Predictive Analytics in Competitive Domains

Leung and Joseph (2014) demonstrated the effectiveness of data mining techniques for predicting outcomes in competitive domains, applying classification analysis to college football games using historical win/loss data. Their approach---combining multiple statistical measures to predict outcomes in a competitive system where individual matchup context matters---provides a methodological template for CCG analysis: in both domains, historical performance data must be contextualized rather than naively aggregated. Their work additionally validates the use of ensemble methods for prediction tasks where multiple weak signals combine to produce meaningful forecasts.

## 2.2 Data Mining in Collectible Card Games

Academic literature on CCG analytics remains sparse; the most directly relevant studies are unpublished student project reports, reflecting the nascent state of this research area. Hoover et al. (2020) provide a systematic overview of AI challenges in Hearthstone, identifying deckbuilding, game balance, and player modeling as key open problems---an observation that motivates the present study's application to Lorcana.

Pawlicki, Polin, and Zhang (2014) applied logistic regression and support vector machines to Magic: The Gathering, using historical price, sales volume, and tournament-usage data to predict short-term price increases. Their finding that tournament usage is a leading indicator of price movement directly motivates our inclusion of co-occurrence-derived centrality features and the Sleeper Card Index.

Fink, Pastel, and Sapra (2015) predicted MTG card strength from card mechanics alone, demonstrating that intrinsic card features---keyword abilities, mana cost, power/toughness---carry predictive signal for competitive viability without requiring historical tournament data. This work directly motivates our TF-IDF-based feature extraction from Lorcana body text, validating the hypothesis that semantic content of card text encodes strategic value that cannot be captured by numerical attributes alone.

## 2.3 Archetype Discovery and Association Rule Mining

Mora et al. (2022) applied clustering and frequent pattern mining to Hearthstone decklists, demonstrating that statistical techniques can recover recognized strategic archetypes from raw deck data without relying on external community labels. Their finding that FP-Growth outperforms Apriori for CCG decklist mining---due to the sparse, high-cardinality nature of the card item space---directly informs our parameter choices. In the CCG context, each decklist functions as a transaction and each card as an item, making association rule mining a natural framework for synergy discovery.

The theoretical foundation for association rule mining was established by Agrawal et al. (1993), who introduced the support-confidence framework for discovering relationships between items in large transactional databases. Han et al. (2000) subsequently proposed the FP-Growth algorithm, which avoids candidate generation and achieves superior performance on high-dimensional item spaces---critical for CCG applications where the item vocabulary spans hundreds of unique cards.

## 2.4 Explainable Machine Learning

Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), a unified framework for interpreting model predictions based on Shapley values from cooperative game theory. SHAP provides both global feature importance rankings and local per-prediction explanations, making it uniquely suited to CCG analysis where understanding *why* a model rates a card highly is as valuable as the rating itself. We employ TreeSHAP for both Random Forest and XGBoost models, enabling direct comparison of feature importance across model architectures.

## 2.5 Graph Analytics and Community Detection

Blondel et al. (2008) introduced the Louvain algorithm for community detection in large networks, which optimizes modularity through iterative local reassignment. We apply Louvain to deck co-occurrence graphs to recover strategic communities that correspond to known competitive archetypes. Deck co-occurrence graphs represent a natural complement to association rule mining: where rules identify individual card packages, graph structure reveals the full topology of strategic relationships across the card pool.

## 2.6 Time-Series Forecasting

Taylor and Letham (2018) introduced Prophet as a practical decomposable forecasting framework for business time-series with strong trend and seasonality components. Its interpretable component structure---separating trend, seasonality, and holiday effects---makes it well-suited to the CCG domain where tournament cycles and set releases create periodic demand shocks.

Our study synthesizes these methodologies into the entirely novel, dual-market ecosystem of Disney Lorcana.

# 3. Methodology

## 3.1 KDD Process Overview

Our methodology adheres to the canonical KDD process as formalized by Fayyad et al. (1996):

1. **Data Selection and Collection** --- Card data from the Lorcana API, tournament decklists from inkdecks.com, and price snapshots from TCGPlayer.com.
2. **Data Preprocessing** --- Missing value imputation, cross-source name matching via fuzzy matching, Google Sheets formula cleanup, reprint deduplication, and rotation filtering.
3. **Feature Engineering** --- Efficiency ratios, keyword flags, TF-IDF text components, and graph centrality scores fused into a unified dataframe.
4. **Data Mining** --- K-Means clustering for archetype discovery; RF, XGBoost (Chen & Guestrin, 2016), Logistic Regression, and Decision Tree classification (with class-weighted variants) for Top-8 prediction; multi-tier classification for placement tier prediction; placement percentile regression; SHAP for feature attribution; chi-squared testing for archetype balance; NetworkX for co-occurrence analysis; FP-Growth for association rules; Prophet for price forecasting.
5. **Evaluation** --- Time-based train/test split, cross-validation, multi-model comparison across seven configurations, ablation studies, and comparison against majority-class baselines.

## 3.2 Rotation Filter

Disney Lorcana implements a rotation system where only cards from Sets 5 and later are legal for sanctioned competitive play. Tournament decklists were initially filtered by the authors to exclude events known to use pre-rotation card pools. A subsequent automated check identified 9 additional decks containing pre-rotation-only cards (from Sets 1--4 with no reprint in later sets) from the Tokyo and Top Cut Games events. These decks were removed to ensure all analyzed decklists reflect the post-rotation card pool, leaving 753 decks across 21 tournaments.

## 3.3 Implementation Environment

The pipeline is implemented in Python within Google Colab. Core libraries include pandas, scikit-learn, xgboost, shap, networkx, python-louvain, mlxtend, prophet, and matplotlib/seaborn for visualization.

# 4. Data Collection and Preprocessing

Table 1 summarizes the three data sources integrated in this study.

| Source | Records | Scope | Period |
|--------|---------|-------|--------|
| Lorcana API | 2,283 (2,065 unique) | 11 sets, full card pool | -- |
| inkdecks.com | 753 decks (44,884 copies) | 21 tournaments, 12 countries | Sep 2025--Mar 2026 |
| TCGPlayer.com | 407 priced cards (9 snapshots) | Top 100 per window | May 2025--Mar 2026 |

## 4.1 Data Sources

**Card Attribute Data.** 2,283 cards were collected from the Lorcana API (api.lorcana-api.com), encompassing 11 released sets. Each record includes name, set, rarity, ink cost, strength, willpower, lore value, inkability, and full ability text. After deduplicating reprints and alternate-art variants (Enchanted/Epic versions with identical gameplay stats), 2,065 unique cards remained for analysis.

**Tournament Decklists.** Decklists were hand-collected from inkdecks.com, a community platform aggregating results from sanctioned competitive events. The dataset was curated to include only Championship Qualifier (CCQ) and Disney Lorcana Challenge (DLC) tournaments---events with significant prize pools attracting top-tier competition. The 21 selected tournaments span 12 countries across 4 continents, with tournament sizes ranging from 176 to 2,052 registered players (four tournaments exceeded 1,800 players). Critically, inkdecks.com captures only top finishers, not all participants. The resulting 753 decks represent elite competitive performance: 156 Top-8, 124 Top-16, 212 Top-32, 245 Top-64, and 16 Top-128 finishers. Each deck contains 60 cards per Lorcana tournament rules. This elite-only sampling means the classification task distinguishes the best from the nearly-best---not winners from losers---which has significant implications for model performance expectations (Section 6.2). The dataset yields 13,588 individual card-rows (44,884 total card copies when accounting for per-card quantities of 1--4).

**Price Data.** Nine periodic price snapshot files from TCGPlayer.com spanning May 2025 through March 2026 capture the top 100 cards by price movement per window. Across all windows, 407 unique cards appear in the price dataset.

## 4.2 Preprocessing

Missing strength and willpower values (undefined for Action and Item card types) were imputed with zero---semantically meaningful absences, not data errors. Card names were normalized via lowercasing and whitespace standardization, achieving a 100% merge match rate (447 exact matches, 3 fuzzy matches at score $\geq$ 85). Google Sheets formula artifacts (IFERROR wrappers around both card names and quantities) were parsed using regex extraction to recover the cached values embedded in each formula string. 32 decks with missing date fields were excluded from the time-based train/test split but retained for unsupervised analysis (clustering, network, association rules).

## 4.3 Exploratory Data Analysis

Initial exploration revealed important structural properties of the dataset. Of the 2,065 unique cards in the post-rotation pool, 408 (20%) appear in at least one competitive decklist---indicating that the competitive meta concentrates on a relatively narrow slice of the available card pool. Core card statistics (Cost, Strength, Willpower, Lore) show right-skewed distributions with Cost concentrated at 1--5 ink. Keyword abilities are deliberately rare: Evasive appears on approximately 9% of all cards, Challenger on 3%, and Singer on 1%. This rarity is key---when these abilities *do* appear in competitive decks, their disproportionate presence (relative to the card pool) is the signal our models learn.

# 5. Feature Engineering

## 5.1 Structural and Keyword Features

Beyond raw card statistics, we engineered the following derived features:

**Ink Efficiency Ratio** = (Strength + Willpower + Lore) / Cost, capturing overall statistical value per resource unit.

**Lore-per-Ink Ratio** = Lore / Cost, isolating win-condition efficiency.

**10 binary keyword flags** extracted via regex from ability text: Evasive, Rush, Bodyguard, Ward, Shift, Challenger, Reckless, Support, Singer, Resist.

## 5.2 TF-IDF Text Features

Card ability text was vectorized using TF-IDF with 500 unigram/bigram features (min_df=2, English stopwords removed), then reduced to 10 SVD components via Latent Semantic Analysis. These components capture latent strategic dimensions not represented by binary keyword flags: card advantage language ("draw," "look," "add to hand"), board control effects ("banish," "return," "chosen"), and resource mechanics ("exert," "ready").

## 5.3 Deck-Level Aggregation

Card-level features were aggregated to deck-level using quantity-weighted means and sums, producing 25 features per deck including total_lore, avg_cost, avg_rarity, lore_efficiency, keyword copy counts, composite scores (aggro_score, control_score, combo_score), and ink_diversity.

## 5.4 Graph Centrality Features

From the deck co-occurrence network (Section 8), PageRank, betweenness centrality, and degree centrality were extracted per card and merged into the feature matrix as a second data fusion step.

# 6. Results: What Wins

**Card Archetypes vs. Deck Archetypes.** In competitive Lorcana discourse, "archetype" typically refers to a deck strategy (e.g., Blurple, Steelsong). Our analysis distinguishes two levels: *deck-level archetypes* (k=4) describe overall deck strategy, while *card-level archetypes* (k=6) describe the strategic role an individual card plays. A competitive deck typically combines cards from 3--5 card archetypes---for example, a Steelsong deck (deck-level) uses Song-type action cards paired with Singer characters to chain combos, often including cards that cheat out Songs or search for Songs to continue the chain, while also incorporating Cheap Threats for early board presence and Evasive Questers for safe lore generation. The pct_ features in our classification model capture this within-deck composition, and SHAP analysis reveals that this *mix* is more predictive of Top-8 placement than any individual card trait.

## 6.1 Archetype Discovery

K-Means clustering on deck-level features identified k=4 natural deck archetypes:

| Archetype | Community | Decks | Top-8% |
|-----------|-----------|-------|--------|
| Evasive Aggro | Blurple/Dogs | 149 | 25.5 |
| Tempo | Amethyst/Steel | 170 | 19.4 |
| Steelsong | Amber/Steel | 339 | 20.6 |
| Challengers | Amethyst/Steel | 95 | 15.8 |

Steelsong is the most popular archetype (45% of decks), while Evasive Aggro (known in the community as "Blurple" for Amethyst/Sapphire builds and "Dogs" for Amber/Emerald aggro) achieves the highest Top-8 rate (25.5%). Critically, archetype-to-Top-8 correlations remain near zero, indicating no single archetype dominates---a first line of evidence for game balance.

Card-level clustering (k=6) recovered six distinct card archetypes: Cheap Threats (1,052 cards), Beefy Statlines (549), Shift Package (167), Evasive Questers (162), Defensive Utility (72), and Support Cards (63).

![Figure 1: Deck Clusters (PCA Projection) with Top-8 Overlay](fig_06_clusters.png)

![Figure 2: Card Cluster Centroids (Normalized)](fig_08_card_centroids.png)



## 6.2 Supervised Classification

A time-based train/test split at March 2026 yielded 562 training decks (18 tournaments) and 159 test decks (3 tournaments). Mutual Information feature selection identified 7 features with MI > 0.01. An important framing note: because our dataset contains only top finishers from major tournaments (Section 4.1), the classification task is inherently difficult---we are distinguishing Top-8 finishers from Top-9-through-128 finishers, all of whom are elite competitive players at tournaments of up to 2,052 participants.

**Cross-Validation (5-fold TimeSeriesSplit):** RF achieved 0.661 $\pm$ 0.071 weighted F1; XGB achieved 0.634 $\pm$ 0.076.

**Test Set (March 2026):** Majority baseline F1 = 0.780; RF F1 = 0.773 (AUC 0.549); XGB F1 = 0.765 (AUC 0.456).

Neither model exceeds the majority-class baseline. The severe class imbalance (135 non-Top-8 vs. 24 Top-8 in the test set) contributes to poor minority class detection: RF correctly identifies only 1 of 24 actual Top-8 decks (4% recall), while XGB identifies 4 of 24 (17% recall).

To address this imbalance, we tested class-weighted variants (class_weight='balanced' for RF, scale_pos_weight for XGB) and two additional baselines (Logistic Regression and Decision Tree, both with balanced class weights):

| Model | W-F1 | AUC | Top-8 Recall |
|-------|------|-----|-------------|
| Majority Baseline | 0.780 | 0.500 | 0/24 |
| Logistic Regression (balanced) | 0.622 | 0.578 | 12/24 |
| Decision Tree (balanced) | 0.438 | 0.420 | 13/24 |
| Random Forest | 0.773 | 0.549 | 1/24 |
| Random Forest (balanced) | 0.817 | 0.518 | 3/24 |
| XGBoost | 0.765 | 0.456 | 4/24 |
| XGBoost (balanced) | 0.769 | 0.505 | 2/24 |

A clear tradeoff emerges: simpler models with aggressive class weighting (LR, Decision Tree) detect more Top-8 decks but at severe cost to overall accuracy, falling well below the majority baseline. Ensemble models (RF, XGB) achieve stronger overall F1 but cannot reliably identify Top-8 decks. No model achieves both competitive overall accuracy and meaningful minority class recall---the strongest evidence that deck composition alone is insufficient to predict tournament outcomes.

![Figure 3: Multi-Model Comparison — Top-8 Classification](fig_multimodel_comparison.png)

To further probe the boundary between deck signal and noise, we extended the analysis beyond binary classification to multi-tier prediction and continuous regression.

Multi-tier classification (Top-8 / Top-16 / Top-32 / Top-64) was also tested using enhanced features including tournament size and set recency, yielding weighted F1 of only 0.347 (RF) and 0.314 (XGB)---near random for four classes, confirming that placement tiers are indistinguishable from deck composition.

However, regression on placement percentile (Placement / Player_Count) reveals a more detailed picture. Using deck composition features only (excluding Player_Count to avoid target leakage, since it appears in the denominator of the target), RF regression achieves R$^2$ = 0.007 with Spearman $\rho = 0.235$ ($p = 0.003$). Deck composition provides a statistically significant but very weak *directional* signal---better-composed decks tend to place slightly higher---but explains less than 1% of placement variance. Over 99% is attributable to player skill, matchup variance, and in-game decision-making. This distinction between directional signal (significant) and practical signal (negligible) is the most precise characterization of game balance our pipeline produces: *deck building provides a slight edge, but player skill overwhelmingly determines outcomes.*

![Figure 4: Confusion Matrices — Test Set (March 2026)](fig_28_confusion_matrices.png)


## 6.3 SHAP Feature Importance

SHAP analysis (Lundberg & Lee, 2017) reveals which features the models *do* learn from, even though they cannot reliably predict individual outcomes. An important methodological note: SHAP values characterize the data structure that models detect, not the models' predictive accuracy. Even a below-baseline model can correctly identify which features competitive decks share; the failure lies in converting those patterns into reliable individual predictions, consistent with a game where deck composition is necessary but not sufficient. Both RF and XGB independently converge on the same top 5 features, reinforcing that these patterns are genuine properties of the data rather than model artifacts.

**Top 5 features:**

| Rank | Feature | RF | XGB |
|------|---------|-----|------|
| 1 | avg_rarity | .025 | .319 |
| 2 | avg_cost | .019 | .554 |
| 3 | pct_cheap_threats | .017 | .329 |
| 4 | pct_beefy_statlines | .014 | .246 |
| 5 | lore_efficiency | .011 | .218 |

Note: RF SHAP values are smaller in magnitude because Random Forest operates on a probability scale (0--1), while XGBoost SHAP values reflect log-odds; rankings, not magnitudes, should be compared across models. Table ranked by RF; both models agree on the same top 5 features in slightly different order.

**Bottom features:** ward_copies, resist_copies, ink_diversity, and bodyguard_copies consistently rank at the bottom of both models (RF: 0.001--0.002; XGB: 0.003--0.023). Evasive_copies and singer_copies also rank in the bottom half---not because they lack competitive value, but because their signal is already captured by the archetype composition features above them.

Top-8 decks lean slightly cheaper: 44.6% of cards at ink cost 1--2 compared to 41.8% for non-Top-8. Quantity-weighted deck stats are nearly identical (Cost: 3.24 vs 3.34; Strength: 1.82 vs 1.87; Willpower: 2.77 vs 2.80; Lore: 1.08 vs 1.07), with ink cost showing the largest gap---reinforcing that efficiency, not raw power, differentiates winning decks. In simpler terms, winning decks play more cheap cards (cost 1--2) and fewer expensive ones, allowing them to deploy more cards per turn and quest for lore faster.

![Figure 5: Feature Importance (SHAP) — RF vs XGBoost](fig_13_shap_comparison.png)



SHAP dependence plots reveal clear nonlinear patterns: for avg_rarity, decks with lower rarity receive positive SHAP values (pushing toward Top-8), with the strongest effects concentrated among the lowest-rarity decks. Similarly, lower avg_cost values push toward Top-8. This confirms that rarity and cost efficiency are the primary optimization axes for competitive deck building.

![Figure 6: SHAP Dependence Plots (XGBoost)](fig_14_dependence_plots.png)

## 6.4 MI vs. SHAP Methodological Comparison

A notable methodological finding emerges from comparing Mutual Information and SHAP rankings. Singer_copies and bodyguard_copies rank high in MI but bottom-half in SHAP (bodyguard at 0.001, near last). This divergence reflects a fundamental difference: MI captures marginal statistical dependence, while SHAP measures each feature's contribution in the context of all other features. The model doesn't use Singer because avg_rarity and avg_cost already capture the same signal more efficiently. Researchers should not rely solely on univariate feature selection when multivariate models are the end goal.

## 6.5 Three-Layer Framework

Synthesizing SHAP, MI, weighted baseline, and association rule findings, we propose a three-layer framework for competitive advantage in Disney Lorcana:

**Layer 1 (Table Stakes).** Evasive and draw mechanics are universally adopted across competitive decks. Their universal presence is itself the strongest signal of necessity---competitive players would not dedicate deck slots to these cards if they were not essential. However, because every Top-8 deck includes them, they carry no *discriminative* signal for prediction. Evasive's mechanical advantage is substantial: only characters with Evasive or the Alert keyword can challenge Evasive characters, making them extremely safe for questing. This mechanical dominance explains their universal adoption.

**Layer 2 (Deck Optimization).** This is where competitive edge lives. The data shows that winning decks have lower average card rarity, a balanced mix of card roles (not too aggro-heavy), and efficient lore-to-cost ratios. In practical terms: build a well-rounded deck with cost-efficient cards rather than stuffing it with expensive rares. avg_rarity, avg_cost, and archetype composition (pct_cheap_threats, pct_beefy_statlines) are the top SHAP features across both models.

**Layer 3 (Card Packages).** Locked pairs from FP-Growth---Aurora -- Holding Court and Tiana -- Restaurant Owner at 100% confidence and lift 15.8, Vincenzo Santorini -- The Explosives Expert and Tinker Bell -- Giant Fairy at 100% confidence and lift 14.9---confirm that competitive success depends on combinatorial deck building, not individual card power. These synergistic packages represent the final optimization layer beyond deck-level composition.

# 7. Results: What Sells

## 7.1 Trait-Price Spearman Correlations

Spearman rank correlations between card traits and market price (124 matched cards) reveal that numeric stats---not keyword abilities---drive price:

| Trait | $\rho$ | p |
|-------|--------|-------|
| Cost | 0.356 | <.001 |
| Lore | 0.216 | .016 |
| Strength | 0.212 | .018 |
| has_evasive | 0.016 | .857 |
| has_singer | -0.105 | .247 |
| has_challenger | -0.155 | .085 |

Cost is the strongest predictor ($\rho = 0.356$), consistent with higher-rarity cards having higher costs. Keyword traits show no significant positive correlation with price---a stark contrast with their competitive importance (SHAP). Notably, challenger cards show a negative price-level correlation but *positive* price change correlation ($\rho = +0.145$, $p = 0.016$), suggesting the market is catching up to their competitive value.

![Figure 7: Spearman Correlation — Card Traits vs. Market Price](fig_17_trait_price_spearman.png)


## 7.2 Price Regression

XGBoost regression on card traits yields negative R-squared across all configurations, confirming that **traits predict competitive presence, not market price**. This is an honest null finding: the secondary market is driven by rarity, Disney IP recognition, and collector demand---factors unrelated to competitive utility.

## 7.3 Collector-Competitor Market Divergence

Only 35% of priced cards see competitive play. Price-change SHAP analysis reveals that weighted_tournament_presence (mean |SHAP| = 0.230) dominates price appreciation predictions, followed by rarity_encoded (0.151). The market does reward competitive success---but through aggregate tournament presence, not individual card traits.

## 7.4 Prophet Forecasting (Proof of Concept)

We applied Prophet (Taylor & Letham, 2018) to cards with at least four price observations across snapshot windows. Prophet was configured with changepoint_prior_scale = 0.3 and all seasonality disabled given the monthly data granularity. Of qualifying cards, the top 5 by data availability were forecast over 60-day horizons. Results are illustrative: Lady -- Miss Park Avenue shows steady appreciation, while Scrooge McDuck -- Resourceful Miser shows sustained depreciation. Donald Duck -- Musketeer (Promo version) exhibits the tightest confidence intervals and strongest uptrend (\$46.78 to \$74.44), consistent with its status as a high-value collector card---notably, Donald Duck -- Musketeer does not appear in competitive decklists, making its price trajectory purely collector-driven and further illustrating the divergence between competitive and collector markets.

These forecasts demonstrate the applicability of Prophet's decomposable framework to the Lorcana market, but meaningful MAPE validation would require daily price data---identified as a priority for future work.

# 8. Results: Network and Synergy Analysis

## 8.1 Co-occurrence Graph

A weighted co-occurrence graph was constructed from decklists with a minimum edge threshold of 15 co-occurrences, producing 152 nodes and 1,855 edges.

**Community Detection.** Louvain community detection (Blondel et al., 2008) identified 4 communities, matching the k=4 deck-level clustering---convergence across two independent methods that strengthens both findings.

**Top cards by PageRank:** Elsa -- The Fifth Spirit (0.030), Genie -- Wish Fulfilled (0.029), Tipo -- Growing Son (0.027), Sail the Azurite Sea (0.026). These are meta-defining competitive staples that appear as universal inclusions across multiple communities.

**Top cards by Betweenness Centrality:** Vision of the Future (0.083), He Hurled His Thunderbolt (0.082), Finnick -- Tiny Terror (0.081), Mowgli -- Man Cub (0.080), and Hades -- Infernal Schemer (0.067). These "bridge" cards connect archetypes, enabling cross-community deck building and providing generalist value across strategic contexts.

![Figure 8: Card Co-occurrence Network (4 Communities Detected)](fig_25_network.png)


## 8.2 Association Rule Mining

FP-Growth (Han et al., 2000) with min_support=0.05 and max_len=2 on the most frequent cards produced 889 frequent itemsets and 1,112 association rules (lift $\geq$ 2.0).

**Top association rules by lift:**

1. Aurora -- Holding Court $\rightarrow$ Tiana -- Restaurant Owner: 100% confidence, lift 15.8
2. Vincenzo Santorini -- The Explosives Expert $\rightarrow$ Tinker Bell -- Giant Fairy: 100% confidence, lift 14.9
3. Goliath -- Clan Leader $\rightarrow$ Vincenzo Santorini -- The Explosives Expert: 73% confidence, lift 12.4

These "locked packages" confirm that competitive deck building in Lorcana is fundamentally combinatorial. When Aurora -- Holding Court appears in a deck, Tiana -- Restaurant Owner is always present---a perfect 100% co-occurrence. This combinatorial structure represents Layer 3 of our competitive advantage framework.

![Figure 9: Association Rules — Support vs. Confidence](fig_26_association_rules.png)


# 9. Sleeper Card Index

The Sleeper Card Index identifies two types of market inefficiency: (1) cards with strong competitive statistics that are underpriced relative to similar cards---the market has not recognized their value yet; and (2) cards that see little competitive play despite having strong statistical profiles---"sleepers" that the meta has overlooked. Using 5-fold cross-validated XGBoost predictions to prevent memorization:

$$\text{SCI}(\text{card}) = \text{Predicted Price} - \text{Observed Price}$$

The SCI model uses the full feature set (numeric traits, TF-IDF components, and centrality features), which provides stronger predictive signal than the trait-only regression reported in Section 7.2.

Rarity abbreviations: L = Legendary, SR = Super Rare, R = Rare, U = Uncommon. Prices from TCGPlayer snapshots (May 2025--March 2026). Dollar values rounded; SCI computed from unrounded values.

**Top 5 Undervalued (Positive SCI):**

1. Enigmatic Inkcaster (R) --- Observed: \$2, Predicted: \$37, SCI: +\$35
2. Prince Phillip -- Vanquisher of Foes (SR) --- Observed: \$1, Predicted: \$26, SCI: +\$25
3. Anna -- Magical Mission (R) --- Observed: \$4, Predicted: \$22, SCI: +\$18
4. Penny -- Bolt's Person (U) --- Observed: \$4, Predicted: \$20, SCI: +\$16
5. Jafar -- Keeper of Secrets (R) --- Observed: \$2, Predicted: \$18, SCI: +\$16

**Top 5 Overvalued (Negative SCI):**

1. Baymax -- Armored Companion (L) --- Observed: \$66, Predicted: \$15, SCI: $-$\$52
2. Cinderella -- Dream Come True (L) --- Observed: \$44, Predicted: \$7, SCI: $-$\$37
3. Angel -- Experiment 624 (L) --- Observed: \$43, Predicted: \$9, SCI: $-$\$34
4. Dumbo -- Ninth Wonder of the Universe (L) --- Observed: \$49, Predicted: \$18, SCI: $-$\$31
5. Demona -- Scourge of the Wyvern Clan (L) --- Observed: \$40, Predicted: \$8, SCI: $-$\$31

The pattern is striking: every overvalued card is Legendary rarity (L), while the undervalued list consists entirely of Rare (R), Super Rare (SR), and Uncommon (U) cards. This confirms that the collector premium is driven primarily by rarity tier and Disney IP recognition rather than competitive utility. One caveat: Baymax -- Armored Companion's observed price (\$66) likely reflects an Enchanted or Promo variant rather than the standard printing (approximately \$2--3 as of April 2026), which would significantly reduce its SCI magnitude. The remaining four overvalued cards reflect standard Legendary pricing.

![Figure 10: Sleeper Card Index — Model Predicted vs. Observed Price](fig_27_sci.png)


# 10. Ablation Studies

## 10.1 Multi-Modal Feature Ablation

An ablation study measures the incremental value of each feature engineering layer for predicting competitive card usage (Quantity as target):

| Config | # | RMSE | Gain |
|--------|---|------|------|
| Numeric Only | 16 | 268.6 | -- |
| + TF-IDF | 26 | 245.8 | +8.5% |
| + TF-IDF + Centrality | 29 | 38.5 | +85.7% |

Each successive layer provides measurable improvement, with network centrality contributing a striking increment. This confirms that co-occurrence patterns---which cards appear together in competitive decks---carry far more information about competitive usage than either numeric traits or text features alone, extending the findings of Pawlicki et al. (2014) and Fink et al. (2015) to the Lorcana ecosystem. A caveat: centrality features are derived from the same decklists used to compute the target variable (Quantity), so part of this improvement reflects shared information rather than independent predictive signal. The centrality layer is best interpreted as confirming the dominance of network-level patterns over card-level traits, rather than as a pure out-of-sample prediction gain.

## 10.2 Clustering Feature Ablation

Comparing XGBoost with and without archetype composition features on the test set:

| Config | F1 |
|--------|------|
| With clustering | 0.765 |
| Without clustering | 0.757 |
| **Difference** | **+0.008** |

Archetype decomposition provides incremental predictive value, justifying the clustering pipeline.

# 11. Discussion

## 11.1 Game Balance Evidence

We operationalize game balance along two dimensions: *meta diversity* (whether multiple strategic approaches are competitively viable) and *skill dominance* (whether player skill outweighs deck construction in determining outcomes). The competitive Lorcana community defines balance primarily through meta diversity---the expectation that all color combinations should be viable. Our analytical framework extends this by quantifying skill dominance through predictive modeling, building on the approach of García-Sánchez et al. (2018), who used evolutionary algorithms to measure game balance in Hearthstone by searching for card attribute changes that equalize win rates across decks. Four independent lines of evidence converge on a single conclusion: Disney Lorcana is a well-balanced competitive game. Importantly, all 753 decks in our dataset are top finishers at major international tournaments (176--2,052 players), so the classification task separates the best from the nearly-best---the hardest possible prediction problem for deck-based models.

1. **Near-zero archetype correlations**: Card archetype composition features show correlations with Top-8 placement ranging from $-0.043$ to $+0.026$---no archetype systematically produces winners. A chi-squared test confirms that archetype Top-8 rates do not differ significantly ($\chi^2 = 3.66$, $p = 0.301$, $df = 3$).
2. **Balanced cluster Top-8 rates** (15.8%--25.5%): All four deck archetypes produce Top-8 finishers.
3. **Fundamental accuracy-recall tradeoff**: Across seven model configurations (including class-weighted variants and simpler baselines), no model exceeds the majority-class baseline while also achieving meaningful Top-8 recall.
4. **Directional but not practical signal**: Placement percentile regression yields Spearman $\rho = 0.235$ ($p = 0.003$) but R$^2$ = 0.007---deck composition provides a statistically significant rank-order signal but explains less than 1% of placement variance. Over 99% remains unexplained by deck composition, attributable to player skill, matchup variance, and in-game decision-making.

The competitive meta spans at least 22 distinct color combinations. The top three---Amethyst/Sapphire (183 decks, 19.1% Top-8 rate), Amethyst/Steel (157 decks, 22.9%), and Amber/Emerald (109 decks, 24.8%)---account for 60% of the field but none dominates Top-8 conversion. This diversity of competitive color combinations provides further evidence of game balance at the strategic level.

Community players observe that successful decks across color combinations converge toward midrange or control play patterns---consistent with our SHAP finding that lower aggro percentage and higher lore efficiency push toward Top-8.

This is evidence of healthy competitive game design---Lorcana rewards strategic deck building as a *necessary but not sufficient* condition for success. The remaining variance is player skill, matchup variance, and in-game adaptation---exactly what competitive game designers aim for. Notably, only 408 of 2,065 cards (20%) appear in any competitive decklist, indicating that the meta concentrates on a narrow slice of the card pool---yet within that slice, no single composition dominates. For context, this level of balance compares favorably to established CCGs: in Hearthstone, even the best decks typically achieve only 53--55% win rates, yet dominant strategies still emerge frequently enough to require developer intervention through card nerfs; in Magic: The Gathering, Standard formats regularly produce decks with 55--60% win rates that necessitate ban-list updates. Lorcana's archetype-level balance (15.8%--25.5% Top-8 rates, $\chi^2$ $p = 0.301$) and the near-zero explanatory power of deck composition ($R^2 = 0.007$) suggest a competitive environment where strategic diversity is not merely present but structurally embedded in the game's design. Even total lore (the primary deck-level win condition metric) is nearly identical across placement tiers: Top-8 decks average 64.1 lore vs. 63.9 for Top-64, confirming that deck composition alone does not separate finishers. Set recency---the proportion of cards from the newest available sets---shows no correlation with placement ($\rho = -0.011$, $p = 0.769$), indicating that adapting to the newest cards does not confer a competitive advantage.

## 11.2 Implications for Competitive Players

**Table Stakes First.** Ensure Evasive and draw mechanics are present in every competitive build. These are non-negotiable---every Top-8 deck includes them.

**Optimize Deck Composition.** Lower average rarity and average cost push toward Top-8. Build a well-rounded deck with cost-efficient cards rather than loading up on expensive rares. The data shows that 44.6% of Top-8 deck cards are cost 1--2, compared to 41.8% in non-Top-8 decks.

**Select Synergistic Packages.** Association rules reveal locked card pairs (e.g., Aurora -- Holding Court and Tiana -- Restaurant Owner at 100% confidence) that form the backbone of competitive archetypes. Identify and include these proven synergy packages rather than selecting cards individually.

**Budget-Friendly Competition.** The SHAP finding that lower avg_rarity pushes toward Top-8 suggests that efficient, budget-conscious builds can compete with expensive decks---good news for players entering the competitive scene.

## 11.3 Implications for Collectors

The SCI distinguishes competitive value from collector premium. Cards with high positive SCI represent strong acquisition targets: strategically validated yet underpriced. Cards with large negative SCI carry Disney IP premiums that may not persist if competitive irrelevance becomes more widely recognized.

## 11.4 Limitations

**Price Data Granularity.** Monthly snapshots limit Prophet forecasting precision and prevent measurement of tournament-to-price lag dynamics.

**Decklist Representativeness.** Tournament sources from inkdecks.com may overrepresent competitive regional metas and underrepresent casual play.

**Meta Non-Stationarity.** Each tournament corresponds to a different active set release, meaning the available card pool expands over time. Cards from Sets 5--11 appear in the dataset, with Sets 5 (2,541 card-rows), 9 (3,045), and 10 (2,695) most heavily represented. Co-occurrence graphs, FP-Growth rules, and PageRank centrality aggregate across all 753 decks, pooling multiple meta snapshots. This is a deliberate design choice---the post-rotation card pool (Sets 5+) provides a stable base, and new set additions are incremental---but readers should note that association rules and centrality rankings reflect the *aggregate* seven-month meta rather than any single point-in-time snapshot. The time-based train/test split partially mitigates this concern for classification, but unsupervised results (clustering, network, rules) represent a meta-averaged view. New set releases will introduce format shifts requiring periodic model recalibration.

**TF-IDF Limitations.** Bag-of-words treatment cannot capture sequential or conditional logic of complex multi-clause card effects.

**Multicollinearity.** Core stat features (Cost, Strength, Willpower, Lore) show moderate correlation, which may inflate individual SHAP values for correlated features. The aggregate pattern (deck optimization > keyword traits) remains stable.

**API Data Quality.** The Lorcana API's Inkable field contains known inaccuracies for certain cards (e.g., Sail the Azurite Sea incorrectly marked as non-inkable), preventing reliable analysis of ink ratio optimization across competitive decks.

**Reprint Effects on SCI.** Cards with recent reprints may appear undervalued in the SCI because the reprint increases supply without changing competitive demand, depressing observed price. The SCI does not account for supply-side dynamics.

**Price Variant Conflation.** TCGPlayer price snapshots may conflate standard and premium card variants (Enchanted, Promo, DLC). At least one card (Baymax -- Armored Companion) appears to reflect premium variant pricing rather than the standard printing, which would reduce its SCI overvaluation.

**Class Imbalance.** Top-8 decks represent approximately 15--20% of the dataset, creating a class imbalance that biases models toward majority-class prediction. Class-weighted model variants were tested: RF balanced improved weighted F1 from 0.773 to 0.817 and Top-8 recall from 4% to 12.5%, but XGB balanced showed no meaningful improvement. Simpler models (Logistic Regression, Decision Tree) with balanced weights achieved up to 50% Top-8 recall but at severe cost to overall accuracy (F1 = 0.622 and 0.438, respectively). The fundamental tradeoff between accuracy and minority recall persists across all configurations, suggesting the weak signal reflects genuine game balance rather than a correctable modeling limitation.

**No Player-Level Features.** Our models predict Top-8 placement from deck composition alone, excluding player skill, seeding, and historical performance. Since our own findings indicate that player skill is the dominant factor in outcomes, incorporating player-level features in future work could separate the deck-construction signal from the player-skill signal more cleanly. A placement_weight variable (scaling from 1.0 for Top-8 to 0.05 for Top-128) was computed but intentionally excluded from classification features to avoid target leakage; future work could explore ordinal regression using placement tiers as a multi-class target.

## 11.5 Summary for Competitive Players

The following restates the key competitive findings from Sections 6--10 in non-technical language intended for the player community.

**Build cheap.** Top-8 decks run more 1-2 cost cards (44.6%) than non-Top-8 decks (41.8%). Lower average rarity is the single strongest predictor of success---you don't need expensive cards to win.

**Run Evasive.** Every competitive deck includes Evasive characters. Only Evasive or Alert characters can challenge them, making them the safest way to quest for lore. This is non-negotiable.

**Pick proven pairs.** Certain cards always appear together---Aurora -- Holding Court with Tiana -- Restaurant Owner (100% of the time), Vincenzo Santorini -- The Explosives Expert with Tinker Bell -- Giant Fairy (100%). Build around these proven synergies rather than selecting cards individually.

**Don't overpay for Legendaries.** Every card on our "overvalued" list is Legendary. Cinderella -- Dream Come True (L, \$44) and Angel -- Experiment 624 (L, \$43) carry Disney IP collector premiums far beyond their competitive value. Budget decks can and do make Top-8.

**No single deck wins.** All four major archetypes (Steelsong, Evasive Aggro, Tempo, Challengers) produce Top-8 finishers. Play what you know best---player skill matters more than deck choice.

## 11.6 Future Work

**Higher-Frequency Price Data.** Daily or weekly price collection would enable validated Prophet forecasting with MAPE evaluation and measurement of tournament-to-price lag dynamics.

**Advanced Text Embeddings.** Transformer-based models (e.g., fine-tuned BERT) could capture richer ability text semantics than the current bag-of-words TF-IDF approach.

**Temporal Archetype Tracking.** Monitoring archetype composition over time would enable concept drift analysis as new sets release and the meta evolves.

**Finer-Grained Classification.** A multi-tier ordinal model (Top 8 / Top 16 / Top 32 / Top 64) could capture competitive signal lost by the current binary Top-8 / non-Top-8 split.

**Generative Deck Optimization.** Reinforcement learning could extend synergy discovery from descriptive analysis to automated deck generation.

**Representative Decklists.** Generating sample decklists per archetype from association rule outputs as seed packages would provide directly actionable guidance for competitive players.

# 12. Conclusion

This paper presented a complete KDD framework applied to Disney Lorcana---to our knowledge, the first academic study of this kind. Integrating 2,065 unique card records, 753 competitive decklists across 21 tournaments, and periodic price snapshots, we constructed a pipeline spanning multi-modal feature engineering, unsupervised archetype discovery, explainable supervised classification, network analysis, association rule mining, and exploratory time-series forecasting.

Our central finding---that what wins and what sells are fundamentally different---has both analytical and practical implications. Addressing RQ1, SHAP identifies deck optimization features (avg_rarity, avg_cost, lore_efficiency) as competitive drivers. Addressing RQ2, the market prices Disney IP recognition and rarity rather than competitive traits; the SCI quantifies this gap card by card. Addressing RQ3, the game's competitive balance, evidenced through four independent methods including placement percentile regression ($\rho = 0.235$, R$^2$ = 0.007), suggests that Lorcana's designers have achieved what every competitive game aspires to: a system where deck building provides a slight edge but player skill overwhelmingly determines outcomes.

The multi-modal ablation study demonstrates that TF-IDF text features (+8.5%) and network centrality (+85.7% total) each contribute measurable predictive signal beyond numeric attributes, validating the engineering complexity of the full pipeline. These methods generalize to any CCG with structured card data, competitive decklists, and secondary market price history.

As an initial exploration, this study relied on available historical snapshots from community sources; a continuation would benefit from daily price collection and complete tournament result capture, enabling validated forecasting and finer-grained placement analysis.

# Acknowledgments

AI writing tools (Claude, Anthropic) were used to assist with drafting, editing, and code debugging during this project. All data collection, analysis, interpretation, and domain expertise were performed by the authors.

# References

Agrawal, R., Imielinski, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. *ACM SIGMOD Record*, 22(2), 207--216.

Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, 2008(10), P10008.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785--794.

Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). From data mining to knowledge discovery in databases. *AI Magazine*, 17(3), 37--54.

Fink, D., Pastel, B., & Sapra, N. (2015). Predicting the strength of Magic: The Gathering cards from card mechanics. *Stanford University CS229 Project Report*.

García-Sánchez, P., Tonda, A., Mora, A. M., Squillero, G., & Merelo, J. J. (2018). Automated playtesting in collectible card games using evolutionary algorithms: A case study in Hearthstone. *Knowledge-Based Systems*, 153, 133--146.

Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. *ACM SIGMOD Record*, 29(2), 1--12.

Hoover, A. K., Togelius, J., Lee, S., & de Mesentier Silva, F. (2020). The many AI challenges of Hearthstone. *KI - Künstliche Intelligenz*, 34(1), 33--43.

Leung, C. K., & Joseph, K. W. (2014). Sports data mining: Predicting results for the college football games. *Procedia Computer Science*, 35, 710--719.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Mora, A. M., Tonda, A., Fernandez-Ares, A. J., & Garcia-Sanchez, P. (2022). Looking for archetypes: Applying game data mining to Hearthstone decks. *Entertainment Computing*, 43, 100498.

Pawlicki, M., Polin, J., & Zhang, J. (2014). Prediction of price increase for Magic: The Gathering cards. *Stanford University CS229 Project Report*.

Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37--45.

Ravensburger. (2023). *Disney Lorcana Trading Card Game: The First Chapter* [Card game].

TCGPlayer. (2025). TCGPlayer marketplace price data [Dataset]. https://www.tcgplayer.com

---

**GitHub Repository:** https://github.com/bowmanar-gvsu/lorcana-kdd-analysis
