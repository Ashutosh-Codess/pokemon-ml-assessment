
import os
import time
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIS_DIR = os.path.join(BASE_DIR, "visuals")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "report")

for d in (DATA_DIR, VIS_DIR, MODELS_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

POKEAPI_BASE = "https://pokeapi.co/api/v2/pokemon"
N_POKEMON = 200


# ---------------------------------------------------------------------------
# PART 1 - Data Collection & Preprocessing
# ---------------------------------------------------------------------------
def fetch_pokemon_record(pokedex_id: int, requests_module) -> dict:
    resp = requests_module.get(f"{POKEAPI_BASE}/{pokedex_id}", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
    primary_type = data["types"][0]["type"]["name"]
    return {
        "name": data["name"],
        "height": data["height"],
        "weight": data["weight"],
        "base_experience": data["base_experience"],
        "hp": stats.get("hp"),
        "attack": stats.get("attack"),
        "defense": stats.get("defense"),
        "special_attack": stats.get("special-attack"),
        "special_defense": stats.get("special-defense"),
        "speed": stats.get("speed"),
        "primary_type": primary_type,
    }


def build_part1_dataset() -> pd.DataFrame:
    print("\n=== PART 1: Data Collection & Preprocessing ===")
    cache_path = os.path.join(DATA_DIR, "pokemon_dataset.csv")
    df = None

    try:
        import requests
        test = requests.get(f"{POKEAPI_BASE}/1", timeout=5)
        test.raise_for_status()
        records = []
        for pokedex_id in range(1, N_POKEMON + 1):
            try:
                records.append(fetch_pokemon_record(pokedex_id, requests))
            except Exception as exc:
                print(f"  Skipping id={pokedex_id}: {exc}")
            time.sleep(0.05)
        df = pd.DataFrame(records)
        print(f"  Fetched {len(df)} records live from PokeAPI.")
    except Exception as e:
        print(f"  Live PokeAPI unreachable ({e!r}); using cached snapshot.")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
        else:
            raise RuntimeError(
                "No live network and no cached dataset found at "
                f"{cache_path}. Cannot proceed."
            )

    base_cols = ["name", "height", "weight", "base_experience", "hp", "attack",
                 "defense", "special_attack", "special_defense", "speed", "primary_type"]
    df = df[base_cols].copy()

    # Clean: dedupe, validate types, handle nulls
    df = df.drop_duplicates(subset="name").reset_index(drop=True)
    numeric_cols = ["height", "weight", "base_experience", "hp", "attack",
                     "defense", "special_attack", "special_defense", "speed"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    if df["primary_type"].isnull().any():
        df["primary_type"] = df["primary_type"].fillna(df["primary_type"].mode()[0])
    df[numeric_cols] = df[numeric_cols].astype(int)

    # Encode categorical column
    le = LabelEncoder()
    df["primary_type_encoded"] = le.fit_transform(df["primary_type"])

    assert df.shape[0] >= 200, "Dataset must contain at least 200 Pokemon"
    assert df.isnull().sum().sum() == 0

    df.to_csv(cache_path, index=False)
    print(f"  Saved: {cache_path}  shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# PART 2 - Feature Engineering & EDA
# ---------------------------------------------------------------------------
def build_part2_eda(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PART 2: Feature Engineering & EDA ===")
    sns.set_theme(style="whitegrid")
    df = df.copy()

    df["total_power"] = (df["hp"] + df["attack"] + df["defense"] +
                          df["special_attack"] + df["special_defense"] + df["speed"])
    median_power = df["total_power"].median()
    df["is_high_power"] = (df["total_power"] > median_power).astype(int)
    print(f"  median(total_power) = {median_power}")

    out_csv = os.path.join(DATA_DIR, "pokemon_feature_engineered.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # Visualization 1: Attack Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="attack", kde=True, bins=20, color="#4C72B0")
    plt.title("Attack Distribution (Histogram + KDE)")
    plt.xlabel("Attack"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "attack_distribution.png"), dpi=150)
    plt.close()

    # Visualization 2: Correlation Heatmap
    numeric_cols = ["height", "weight", "base_experience", "hp", "attack",
                     "defense", "special_attack", "special_defense", "speed", "total_power"]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"label": "Correlation"})
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # Visualization 3: Type Frequency
    plt.figure(figsize=(11, 6))
    type_counts = df["primary_type"].value_counts()
    sns.barplot(x=type_counts.values, y=type_counts.index, hue=type_counts.index,
                palette="viridis", legend=False)
    plt.title("Pokemon Primary Type Frequency")
    plt.xlabel("Count"); plt.ylabel("Primary Type")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "type_frequency.png"), dpi=150)
    plt.close()

    # Visualization 4: Speed vs Attack
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="attack", y="speed", hue="is_high_power",
                     palette={0: "#4C72B0", 1: "#DD8452"}, alpha=0.8)
    plt.title("Speed vs Attack (colored by High Power status)")
    plt.xlabel("Attack"); plt.ylabel("Speed")
    plt.legend(title="is_high_power")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "speed_vs_attack.png"), dpi=150)
    plt.close()

    print("  Saved 4 visualizations to visuals/")
    return df


# ---------------------------------------------------------------------------
# PART 3 - Regression
# ---------------------------------------------------------------------------
def build_part3_regression(df: pd.DataFrame):
    print("\n=== PART 3: Regression (predict total_power) ===")
    feature_cols = ["hp", "attack", "defense", "special_attack", "special_defense",
                     "speed", "height", "weight", "base_experience"]
    X, y = df[feature_cols], df["total_power"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(mae, 6), "MSE": round(mse, 6), "RMSE": round(rmse, 6),
        "R2_Score": round(r2, 6), "Mean_CV_R2": round(cv_scores.mean(), 6),
    }
    pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_csv(
        os.path.join(REPORT_DIR, "regression_metrics.csv"), index=False)
    joblib.dump(model, os.path.join(MODELS_DIR, "linear_regression.pkl"))

    print(f"  Test metrics: {metrics}")
    print("  Saved: models/linear_regression.pkl, report/regression_metrics.csv")
    return model, metrics


# ---------------------------------------------------------------------------
# PART 4 - Classification & GridSearchCV
# ---------------------------------------------------------------------------
def build_part4_classification(df: pd.DataFrame):
    print("\n=== PART 4: Classification & GridSearchCV (predict is_high_power) ===")
    feature_cols = ["hp", "attack", "defense", "special_attack", "special_defense",
                     "speed", "height", "weight", "base_experience"]
    X, y = df[feature_cols], df["is_high_power"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    lr_metrics = (
        accuracy_score(y_test, y_pred_lr), precision_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_lr), f1_score(y_test, y_pred_lr),
        confusion_matrix(y_test, y_pred_lr),
    )

    # Decision Tree (default)
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    dt_metrics = (
        accuracy_score(y_test, y_pred_dt), precision_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_dt), f1_score(y_test, y_pred_dt),
        confusion_matrix(y_test, y_pred_dt),
    )

    # GridSearchCV tuning
    param_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid,
                                cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_dt = grid_search.best_estimator_
    y_pred_tuned = best_dt.predict(X_test)
    tuned_metrics = (
        accuracy_score(y_test, y_pred_tuned), precision_score(y_test, y_pred_tuned),
        recall_score(y_test, y_pred_tuned), f1_score(y_test, y_pred_tuned),
        confusion_matrix(y_test, y_pred_tuned),
    )
    print(f"  Best GridSearchCV params: {grid_search.best_params_}")

    # Confusion matrix plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (acc, prec, rec, f1, cm), title in zip(
        axes, [lr_metrics, dt_metrics, tuned_metrics],
        ["Logistic Regression", "Decision Tree (Default)", "Decision Tree (Tuned - GridSearchCV)"]
    ):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Power", "High Power"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree (Default)", "Decision Tree (Tuned - GridSearchCV)"],
        "Accuracy": [lr_metrics[0], dt_metrics[0], tuned_metrics[0]],
        "Precision": [lr_metrics[1], dt_metrics[1], tuned_metrics[1]],
        "Recall": [lr_metrics[2], dt_metrics[2], tuned_metrics[2]],
        "F1_Score": [lr_metrics[3], dt_metrics[3], tuned_metrics[3]],
    })
    comparison.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)

    joblib.dump(log_reg, os.path.join(MODELS_DIR, "pokemon_classification_model.pkl"))
    joblib.dump(best_dt, os.path.join(MODELS_DIR, "tuned_pokemon_dt_model.pkl"))

    print("  Saved: models/pokemon_classification_model.pkl, models/tuned_pokemon_dt_model.pkl")
    print("  Saved: visuals/confusion_matrix.png, report/model_comparison.csv")
    print(comparison.to_string(index=False))
    return comparison


def main():
    df_raw = build_part1_dataset()
    df_engineered = build_part2_eda(df_raw)
    build_part3_regression(df_engineered)
    build_part4_classification(df_engineered)
    print("\nPipeline complete. All deliverables regenerated under "
          "data/, visuals/, models/, and report/.")


if __name__ == "__main__":
    main()
