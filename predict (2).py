"""Predict production for new rows: python src/predict.py --model outputs/best_model.joblib --input new.csv"""
import argparse, re
import joblib, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="outputs/best_model.joblib")
ap.add_argument("--input", required=True)
a = ap.parse_args()
df = pd.read_csv(a.input)
df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_") for c in df.columns]
df["predicted_production"] = np.expm1(joblib.load(a.model).predict(df)).round(1)
print(df.to_string(index=False))
