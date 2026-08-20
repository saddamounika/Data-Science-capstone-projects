# Bank Marketing — Term Deposit Subscription Prediction

## Problem Statement
This project analyzes direct marketing campaigns (phone calls) of a Portuguese banking
institution to predict whether a client will subscribe to a term deposit. An accurate model
helps the bank optimize future campaigns, reduce client call fatigue, and target
high-propensity customer segments.

## Dataset
Client demographic, campaign, and macroeconomic features (age, job, education, campaign
contact count, `euribor3m`, employment variation rate, etc.), with a binary subscription
outcome (`y`) as the target.

## Key EDA Insight
**Severe class imbalance** — a large majority of clients decline (`"no"`), a small minority
subscribe (`"yes"`). This drove the choice of stratified splitting and class-weight balancing
throughout modeling, and ROC-AUC/F1 over raw accuracy as the primary evaluation metrics.

## A Critical Preprocessing Decision: Dropping `duration`
The dataset documentation states `duration` (the call length) is only known **after** a call
ends — using it as a feature would leak future information into a model meant to plan
*upcoming* campaigns. Despite `duration` being strongly correlated with the outcome, it was
**deliberately excluded** from training features to keep the model usable for real
campaign planning, not just retrospective analysis.

## Approach
1. Preserved `"unknown"` categorical labels (in `education`, `default`, `housing`) as valid
   states rather than dropping them, to maintain data integrity.
2. One-hot encoded nominal attributes; passed macroeconomic indicators through directly to
   preserve their precise metric values.
3. Trained and compared a class-balanced, cross-validated **Tuned Random Forest** against a
   **Gradient Boosting** classifier.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 90.1% | 0.673 | 0.237 | 0.351 | **0.809** |
| Tuned Random Forest | 88.9% | 0.508 | 0.533 | **0.521** | 0.803 |

**Recommended model: Tuned Random Forest** (with cross-validation and class balancing) —
despite Gradient Boosting's higher raw accuracy and ROC-AUC, Random Forest achieves a much
better F1-score and recall. Given the business goal is *finding* likely subscribers (not
just being accurate on the majority "no" class), the higher recall matters more than a
small ROC-AUC edge. Tree-based ensembles also handle this dataset's mix of categorical
attributes and non-linear economic thresholds well, without requiring feature
standardization.

## Actionable Business Recommendations
- **Target high-yield demographics**: student and retired segments show higher baseline
  conversion rates — prioritize these in outbound campaigns.
- **Time campaigns around macroeconomic stability**: `euribor3m` and employment variation
  rate strongly affect subscription likelihood.
- **Cap repeat contact attempts**: the `campaign` (contact count) feature shows diminishing
  returns and rising customer friction beyond a few calls — recommend capping at 3 attempts.

## Tools Used
`Python` · `pandas` · `scikit-learn` (Random Forest, Gradient Boosting, OneHotEncoder,
cross-validation)

## How to Reproduce
Open `Protuguese_bank_marketing.ipynb` and run all cells top to bottom.
