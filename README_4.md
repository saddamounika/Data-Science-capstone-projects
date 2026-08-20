# Cellphone Price Range Prediction

## Problem Statement
A mobile phone company (fictional client "Bob's Mobile Co.") wants to estimate the price
range of its phones based on technical specifications, in order to compete effectively
against established brands like Apple and Samsung. Rather than predicting an exact price,
the task is to predict a **price range category** — low, medium, high, or very high cost —
from a phone's hardware specifications.

## Dataset
Mobile phone specification data with the following features:
- **Hardware**: `battery_power`, `clock_speed`, `n_cores`, `ram`, `int_memory`, `m_dep`,
  `mobile_wt`
- **Connectivity**: `blue` (Bluetooth), `dual_sim`, `four_g`, `three_g`, `wifi`,
  `touch_screen`
- **Camera & display**: `fc` (front camera MP), `pc` (primary camera MP), `px_height`,
  `px_width`, `sc_h`, `sc_w`
- **Other**: `talk_time`
- **Target**: `price_range` — 0 (low cost), 1 (medium cost), 2 (high cost), 3 (very high cost)

## Approach
1. **Data analysis report** — explored feature distributions and correlations with
   `price_range`, identifying which specifications (notably `ram`, `battery_power`, and
   pixel resolution) most strongly separate price tiers.
2. **Modeling** — trained and compared multiple classifiers, then tuned the top performers
   using `RandomizedSearchCV` for hyperparameter optimization.
3. **Model comparison report** — evaluated all models on accuracy and F1-score to
   recommend a production-ready choice.
4. **Feature importance analysis** — identified which specifications most influence price
   range, to inform both the pricing model and product design decisions.

## Results

| Model | Accuracy | F1-Score |
|---|---|---|
| **Best tuned model** | **93.75%** | **0.937** |

Best result achieved via `RandomizedSearchCV`-tuned Random Forest / Gradient Boosting
classifiers.

*(Full per-model comparison table available in the notebook.)*

## Business Insight
Feature importance analysis highlights which specifications (e.g. RAM, battery capacity,
screen resolution) most strongly drive price positioning — giving the business concrete
guidance on which components to prioritize when designing phones for a target price
segment, rather than pricing reactively after the fact.

## Tools Used
`Python` · `pandas` · `scikit-learn` (RandomizedSearchCV, Random Forest, Gradient Boosting)
· `matplotlib`

## How to Reproduce
Open the notebook and run all cells top to bottom.

---
*Note: This README will be finalized with exact figures and model details once the
completed notebook is added to this folder.*
