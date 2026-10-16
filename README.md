# Project 4 – Prediction of Agriculture Crop Production in India

**Machine Learning Internship – UCT** · Area: Agriculture

## 1. Company
UCT – machine-learning internship programme covering agriculture, predictive maintenance, smart city and industrial manufacturing.

## 2. Background
India is the second most populous country (1.3 billion+) and agriculture is the main livelihood for a large share of its people. Production varies strongly with crop, season, location and cultivated area. The dataset (2001–2014, from data.gov.in, openly licensed) records crop cultivation and production across states.

## 3. Problem statement and relevance
Predict the **production** of a crop given crop, state/zone, season, year and cultivated area (and cost where available). Reliable estimates help farmers choose crops, help planners with procurement/storage, and flag shortfalls early.

## 4. Design
```
raw CSV → cleaning → feature prep → ColumnTransformer → model → evaluation → saved pipeline
                    (log1p target)  (impute+scale | one-hot)   (Ridge / RF / GBM)
```
* **Target:** `production`, modelled as `log1p(production)` because it is extremely right-skewed (sugarcane vs. sesame differ by orders of magnitude). Metrics are reported both in log space and in original units.
* **Features:** numeric (area/quantity, year, cost) and categorical (state, district, season, crop, variety, recommended zone). Column names are auto-detected, so both the data.gov.in layout and the brief's layout work.
* **Models compared:** Ridge (baseline), Random Forest, Gradient Boosting, tuned with 3-fold `GridSearchCV`; 80/20 train/test split, seed 42.

## 5. Implementation
| File | Purpose |
|---|---|
| `src/train.py` | Cleaning, preprocessing, training, tuning, metrics, plots, saves `outputs/best_model.joblib` |
| `src/predict.py` | Predict production for new rows from a CSV |
| `src/make_synthetic_data.py` | Synthetic data for smoke-testing only – **not** used for reported results |

Run:
```bash
pip install -r requirements.txt
python src/train.py --data data/crop_production.csv
python src/predict.py --input new_rows.csv
```
Place the dataset (Google Drive link in the internship brief) at `data/crop_production.csv`.

## 6. Results
> Fill in after running on the real dataset (`outputs/metrics.json`, plots in `outputs/`).

| Model | R² (log) | R² | RMSE | MAE |
|---|---|---|---|---|
| Ridge | _ | _ | _ | _ |
| Random Forest | _ | _ | _ | _ |
| Gradient Boosting | _ | _ | _ | _ |

Plots: `outputs/actual_vs_predicted.png`, `outputs/feature_importance.png`, `outputs/model_comparison.png`.

## 7. Learnings
* Log-transforming a heavy-tailed target is essential; raw-unit R² is dominated by a few very large crops.
* Tree ensembles handle the mixed categorical/numeric data and interactions (crop × area) far better than a linear baseline.
* Rows with missing production must be dropped, not imputed, to avoid training on fabricated targets.
* Limitations: no weather/rainfall/soil data; a random split can leak the same district-year across train and test – a year-based split would be a stricter test.
* Future work: add rainfall/temperature, try CatBoost/LightGBM, hold out whole years.
