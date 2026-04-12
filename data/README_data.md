# Data Sources

## Primary Data Files

### lorcana_master_summary.xlsx
Combined dataset used by the notebook. Contains card attributes, tournament decklists, and price data merged through the preprocessing pipeline.

- **Cards:** 2,652 entries from Lorcast API (2,075 after deduplication)
- **Decks:** 749 tournament decklists from inkdecks.com
- **Prices:** Three sources merged (Lorcast daily, tcgcsv.com market, periodic snapshots)

## Data Collection Details

### Card Data — Lorcast API
- **URL:** https://lorcast.com
- **Coverage:** All Disney Lorcana cards (Sets 1–10 at time of collection)
- **Fields:** Name, ink color, cost, strength, willpower, lore, rarity, abilities, classifications, type, set
- **Limitation:** Reports only primary ink for dual-ink cards (102 of 121 patched via deck context)
- **Daily listing prices:** 2,262 cards with Lorcast-sourced prices

### Tournament Decklists — inkdecks.com
- **URL:** https://inkdecks.com
- **Collection method:** Hand-collected (manual copy from published results)
- **Coverage:** 21 official Disney Lorcana Challenge (DLC) and Community Championship Qualifier (CCQ) events
- **Date range:** September 2025 – March 2026
- **Countries:** 13 (US, UK, Germany, France, Italy, Spain, Australia, Japan, Singapore, Hong Kong, Taiwan, Belgium, Poland)
- **Players:** 710 unique (95.5% single-event)
- **Filtering:** 807 raw → 749 after removing incomplete exports (<60 cards, <5 unique cards)

### Market Prices — tcgcsv.com
- **URL:** https://tcgcsv.com
- **Snapshot date:** April 10, 2026
- **Coverage:** 2,906 price entries
- **Used for:** Primary price source for regression and SCI calculations

### Periodic Price Snapshots
- **Source:** Manual collection from tcgcsv.com "top movers" reports
- **Coverage:** 17 monthly windows (May 2025 – March 2026)
- **Used for:** Temporal price reconstruction (99.8% coverage via stable-card backfill)
- **Method:** Cards absent from "top movers" lists were assigned current prices (stable assumption)

## Data Quality Notes

- **Fuzzy matching:** 448 of 449 tournament card names matched to API names
- **Dual-ink patching:** 102 of 121 dual-ink cards recovered; 19 remain unpatched
- **Price coverage:** 99.8% across temporal windows after backfill
- **Epic rarity:** Introduced after data collection window; not represented in this dataset
- **tcgcsv snapshot:** Single point-in-time; does not capture daily volatility
