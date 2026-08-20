# Flight Fare Prediction

## Problem Statement
Flight ticket prices are highly volatile and unpredictable. This project analyzes historical
flight data to build a model that estimates flight ticket prices — helping customers plan
journeys and airlines understand pricing patterns.

## Dataset
10,683 historical flight records with 11 raw columns: Airline, Date of Journey, Source,
Destination, Departure/Arrival Time, Duration, Total Stops, Additional Info, and Price
(target).

## Approach
1. **Data cleaning** — handled missing values in `Route` and `Total_Stops`.
2. **Feature engineering** (the core challenge in this dataset):
   - Split messy time strings (e.g. `"01:10 22 Mar"`) into clean numerical Hour/Minute columns
   - Converted inconsistent duration formats (`"2h 50m"`, `"19h"`, `"45m"`) into a single
     uniform `Duration_Mins` numeric column via a custom parsing function
   - Unified duplicate location names (`"Delhi"` vs `"New Delhi"`) before one-hot encoding
   - Expanded from 11 raw columns to 30 model-ready features via encoding
3. **Modeling** — trained and compared three tree-based regressors: Decision Tree, Extra
   Trees, and Random Forest.

## Results

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| **Random Forest** | **0.8171** | **₹1,171.04** | ₹1,985.71 |
| Extra Trees | 0.8032 | ₹1,229.88 | ₹2,060.06 |
| Decision Tree | 0.7766 | ₹1,293.86 | ₹2,194.71 |

**Recommended model: Random Forest Regressor** — highest R² (81.71%) and lowest average
error (MAE ₹1,171). As an ensemble of many decision trees, it reduces overfitting and gives
more stable predictions on unseen data than a single Decision Tree.

## Key Challenges & Techniques
- **Complex time strings**: solved with pandas datetime parsing to extract clean numeric
  Hour/Minute features.
- **Irregular duration formats**: solved with a custom Python function to normalize every
  format into total minutes.
- **Duplicate location names**: unified before encoding to avoid the model treating
  "Delhi" and "New Delhi" as separate categories.

## Tools Used
`Python` · `pandas` · `scikit-learn` (Decision Tree, Extra Trees, Random Forest)

## How to Reproduce
Open `flight_fare_prediction.ipynb` and run all cells top to bottom.
