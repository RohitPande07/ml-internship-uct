"""Project 4 - Prediction of Agriculture Crop Production in India (2001-2014).

Usage:
    python src/train.py --data data/crop_production.csv
Column names are auto-detected (case-insensitive) so it works with the
data.gov.in style file (State_Name, District_Name, Crop_Year, Season, Crop,
Area, Production) as well as the column names in the project brief
(Crop, Variety, State, Quantity, Season, Unit, Cost, Recommended Zone).
"""
import argparse, json, re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def norm(c):
    return re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")


def load(path):
    df = pd.read_csv(path)
    df.columns = [norm(c) for c in df.columns]
    return df


def pick(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def prepare(df):
    target = pick(df, "production", "production_tons", "production_in_tons")
    if target is None:
        raise SystemExit(f"No production column found. Columns: {list(df.columns)}")
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df = df[df[target] >= 0]

    num = [c for c in [pick(df, "area", "quantity", "area_hectares"),
                       pick(df, "crop_year", "year"),
                       pick(df, "cost")] if c]
    cat = [c for c in [pick(df, "state_name", "state"), pick(df, "district_name", "district"),
                       pick(df, "season"), pick(df, "crop"), pick(df, "variety"),
                       pick(df, "recommended_zone")] if c]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat:
        df[c] = df[c].astype(str).str.strip()
    # Heavy right-skew in production -> model log1p(production)
    y = np.log1p(df[target])
    return df[num + cat], y, df[target], num, cat, target


def build(model, num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(pipe, X, y_log, y_true, cap=None):
    raw = pipe.predict(X)
    if cap is not None:  # guard against linear-model extrapolation blowing up expm1
        raw = np.clip(raw, 0, cap)
    p = np.expm1(raw)
    return {"R2_log": r2_score(y_log, np.log1p(p.clip(0))),
            "R2": r2_score(y_true, p),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, p))),
            "MAE": float(mean_absolute_error(y_true, p))}, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = load(a.data)
    print(f"Loaded {df.shape[0]} rows, columns: {list(df.columns)}")
    X, y_log, y_raw, num, cat, target = prepare(df)
    print(f"Target={target} | numeric={num} | categorical={cat}")
    Xtr, Xte, ytr, yte, rtr, rte = train_test_split(X, y_log, y_raw, test_size=0.2, random_state=a.seed)

    models = {
        "Ridge": (Ridge(alpha=1.0), {}),
        "RandomForest": (RandomForestRegressor(random_state=a.seed, n_jobs=-1),
                         {"model__n_estimators": [100, 200], "model__max_depth": [None, 20]}),
        "GradientBoosting": (GradientBoostingRegressor(random_state=a.seed),
                             {"model__n_estimators": [150, 300], "model__learning_rate": [0.05, 0.1],
                              "model__max_depth": [3, 5]}),
    }
    results, best_name, best_pipe, best_r2 = {}, None, None, -np.inf
    for name, (m, grid) in models.items():
        pipe = build(m, num, cat)
        if grid:
            pipe = GridSearchCV(pipe, grid, cv=3, scoring="r2", n_jobs=-1).fit(Xtr, ytr).best_estimator_
        else:
            pipe.fit(Xtr, ytr)
        metrics, pred = evaluate(pipe, Xte, yte, rte, cap=ytr.max() + 1)
        results[name] = metrics
        print(name, {k: round(v, 3) for k, v in metrics.items()})
        if metrics["R2_log"] > best_r2:
            best_name, best_pipe, best_r2, best_pred = name, pipe, metrics["R2_log"], pred

    (out / "metrics.json").write_text(json.dumps({"best": best_name, "results": results}, indent=2))
    joblib.dump(best_pipe, out / "best_model.joblib")

    # Plots
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(np.log1p(rte), np.log1p(best_pred.clip(0)), s=4, alpha=.4)
    lim = [0, max(np.log1p(rte).max(), np.log1p(best_pred.clip(0)).max())]
    ax.plot(lim, lim, "r--"); ax.set(xlabel="Actual log(1+production)", ylabel="Predicted",
                                      title=f"{best_name}: actual vs predicted")
    fig.tight_layout(); fig.savefig(out / "actual_vs_predicted.png", dpi=150); plt.close(fig)

    names = best_pipe.named_steps["pre"].get_feature_names_out()
    mdl = best_pipe.named_steps["model"]
    imp = getattr(mdl, "feature_importances_", None)
    if imp is None:
        imp = np.abs(mdl.coef_)
    top = pd.Series(imp, index=names).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5)); top.plot.barh(ax=ax)
    ax.set_title("Top 15 features"); fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150); plt.close(fig)

    # Model comparison
    cmp = pd.DataFrame(results).T
    fig, ax = plt.subplots(figsize=(6, 4)); cmp["R2_log"].plot.bar(ax=ax)
    ax.set_ylabel("R2 (log scale)"); ax.set_title("Model comparison"); fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150); plt.close(fig)
    print(f"Best model: {best_name}. Outputs in {out}/")


if __name__ == "__main__":
    main()
