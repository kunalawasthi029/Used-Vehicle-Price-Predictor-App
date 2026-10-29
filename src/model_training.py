"""
model_training.py - Train and compare ML models for vehicle price prediction.

Trains Linear Regression, Random Forest, and XGBoost regressors,
evaluates their performance, generates comparison plots, and saves
the best-performing model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.data_preprocessing import load_and_preprocess
from src.utils import evaluate_model, save_artifact, PLOTS_DIR, DATA_PATH


# -----------------------------------------------
# Plot styling
# -----------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"lr": "#4C72B0", "rf": "#55A868", "xgb": "#C44E52"}


# -----------------------------------------------
# EDA Plots
# -----------------------------------------------
def generate_eda_plots(filepath=None):
    """Generate and save exploratory data analysis visualizations.

    Produces the following PNG files in the plots/ directory:
        - price_distribution.png
        - price_by_manufacturer.png
        - price_vs_odometer.png
        - price_by_type.png
        - correlation_heatmap.png
        - price_by_fuel.png

    Args:
        filepath: Path to the raw CSV. Defaults to DATA_PATH.
    """
    if filepath is None:
        filepath = DATA_PATH

    print("\n[INFO] Generating EDA plots...")
    df = pd.read_csv(filepath)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Price Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["price"], bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_title("Price Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(
        df["price"].mean(), color="#C44E52", linestyle="--",
        label=f'Mean: ${df["price"].mean():,.0f}',
    )
    axes[0].axvline(
        df["price"].median(), color="#55A868", linestyle="--",
        label=f'Median: ${df["price"].median():,.0f}',
    )
    axes[0].legend()

    sns.boxplot(y=df["price"], ax=axes[1], color="#4C72B0", width=0.4)
    axes[1].set_title("Price Boxplot", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Price ($)")

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "price_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] price_distribution.png")

    # 2. Price by Manufacturer (top 15)
    top_mfr = df["manufacturer"].value_counts().head(15).index
    fig, ax = plt.subplots(figsize=(14, 6))
    order = (
        df[df["manufacturer"].isin(top_mfr)]
        .groupby("manufacturer")["price"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(
        data=df[df["manufacturer"].isin(top_mfr)],
        x="manufacturer", y="price", order=order, palette="coolwarm", ax=ax,
    )
    ax.set_title("Price by Manufacturer (Top 15)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Manufacturer")
    ax.set_ylabel("Price ($)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "price_by_manufacturer.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] price_by_manufacturer.png")

    # 3. Price vs Odometer
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = df.sample(min(5000, len(df)), random_state=42)
    ax.scatter(sample["odometer"], sample["price"], alpha=0.3, s=10, c="#4C72B0")
    ax.set_title("Price vs Odometer", fontsize=14, fontweight="bold")
    ax.set_xlabel("Odometer (miles)")
    ax.set_ylabel("Price ($)")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "price_vs_odometer.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] price_vs_odometer.png")

    # 4. Price by Vehicle Type
    fig, ax = plt.subplots(figsize=(12, 6))
    type_order = df.groupby("type")["price"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="type", y="price", order=type_order, palette="viridis", ax=ax)
    ax.set_title("Price by Vehicle Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Type")
    ax.set_ylabel("Price ($)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "price_by_type.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] price_by_type.png")

    # 5. Correlation Heatmap (numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).drop(
        columns=["id", "price_per_mile"], errors="ignore"
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_cols.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.5, ax=ax, square=True,
    )
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] correlation_heatmap.png")

    # 6. Price by Fuel Type
    fig, ax = plt.subplots(figsize=(10, 6))
    fuel_order = df.groupby("fuel")["price"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="fuel", y="price", order=fuel_order, palette="Set2", ax=ax)
    ax.set_title("Price by Fuel Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fuel")
    ax.set_ylabel("Price ($)")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "price_by_fuel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] price_by_fuel.png")

    print("  All EDA plots saved to plots/\n")


# -----------------------------------------------
# Model Training
# -----------------------------------------------
def train_all_models(X_train, X_test, y_train, y_test, feature_names):
    """Train Linear Regression, Random Forest, and XGBoost models.

    Args:
        X_train, X_test: Feature DataFrames.
        y_train, y_test: Target Series.
        feature_names: List of feature column names.

    Returns:
        list[dict]: Each dict contains 'model_obj', 'model', 'mae', 'rmse', 'r2'.
    """
    results = []

    # 1. Linear Regression
    print("\n[TRAIN] Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_metrics = evaluate_model(y_test, lr_pred, "Linear Regression")
    results.append({"model_obj": lr, **lr_metrics})

    # 2. Random Forest
    print("\n[TRAIN] Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")
    results.append({"model_obj": rf, **rf_metrics})

    # 3. XGBoost
    print("\n[TRAIN] XGBoost...")
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_pred, "XGBoost")
    results.append({"model_obj": xgb, **xgb_metrics})

    return results


# -----------------------------------------------
# Visualization helpers
# -----------------------------------------------
def plot_model_comparison(results):
    """Generate a bar chart comparing MAE, RMSE, and R2 across models.

    Args:
        results: List of metric dicts from train_all_models().
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_names = [r["model"] for r in results]
    mae_vals = [r["mae"] for r in results]
    rmse_vals = [r["rmse"] for r in results]
    r2_vals = [r["r2"] for r in results]
    colors = [COLORS["lr"], COLORS["rf"], COLORS["xgb"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # MAE
    bars = axes[0].bar(model_names, mae_vals, color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("Mean Absolute Error (MAE)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("MAE ($)")
    for bar, val in zip(bars, mae_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold",
        )

    # RMSE
    bars = axes[1].bar(model_names, rmse_vals, color=colors, edgecolor="white", width=0.5)
    axes[1].set_title("Root Mean Squared Error (RMSE)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("RMSE ($)")
    for bar, val in zip(bars, rmse_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold",
        )

    # R2
    bars = axes[2].bar(model_names, r2_vals, color=colors, edgecolor="white", width=0.5)
    axes[2].set_title("R2 Score", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("R2")
    axes[2].set_ylim(0, 1.05)
    for bar, val in zip(bars, r2_vals):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", fontsize=10, fontweight="bold",
        )

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] model_comparison.png")


def plot_feature_importance(model, feature_names, model_name="XGBoost"):
    """Plot horizontal bar chart of feature importances.

    Args:
        model: A fitted model with a ``feature_importances_`` attribute.
        feature_names: Ordered list of feature column names.
        model_name: Label for the plot title.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        range(len(importances)), importances[indices],
        color="#C44E52", edgecolor="white", alpha=0.85,
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Feature Importance ({model_name})", fontsize=14, fontweight="bold")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] feature_importance.png")


def plot_predictions_vs_actual(y_test, y_pred, model_name="XGBoost"):
    """Scatter plot of predicted vs actual prices.

    Args:
        y_test: True target values.
        y_pred: Predicted target values.
        model_name: Label for the plot title.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(y_test), min(3000, len(y_test)), replace=False)
    y_test_arr = np.array(y_test)

    ax.scatter(y_test_arr[sample_idx], y_pred[sample_idx], alpha=0.3, s=10, c="#4C72B0")
    lims = [0, max(y_test_arr.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "--", color="#C44E52", linewidth=2, label="Perfect Prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Price ($)", fontsize=12)
    ax.set_ylabel("Predicted Price ($)", fontsize=12)
    ax.set_title(f"Predicted vs Actual ({model_name})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "predicted_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] predicted_vs_actual.png")


# -----------------------------------------------
# Main pipeline
# -----------------------------------------------
def run_pipeline():
    """Execute the full training pipeline end-to-end.

    Sequence:
        1. Generate EDA plots from raw data
        2. Preprocess data and encode features
        3. Train three regression models
        4. Select the best model by R2 score
        5. Save the best model and evaluation metrics
        6. Generate comparison and diagnostic plots
    """
    print("=" * 55)
    print("  Vehicle Price Prediction - Training Pipeline")
    print("=" * 55)

    # Step 1: EDA visualizations
    generate_eda_plots()

    # Step 2: Data preprocessing
    X_train, X_test, y_train, y_test, encoders, feature_names = load_and_preprocess()

    # Step 3: Train models
    results = train_all_models(X_train, X_test, y_train, y_test, feature_names)

    # Step 4: Identify best model by R2 score
    best = max(results, key=lambda r: r["r2"])
    print(f"\n[BEST] {best['model']}  (R2 = {best['r2']:.4f})")

    # Step 5: Save best model and metrics
    save_artifact(best["model_obj"], "xgboost_best_model.pkl")
    metrics_summary = [
        {"model": r["model"], "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]}
        for r in results
    ]
    save_artifact(metrics_summary, "model_metrics.pkl")

    # Step 6: Generate plots
    plot_model_comparison(results)
    plot_feature_importance(best["model_obj"], feature_names, best["model"])

    best_pred = best["model_obj"].predict(X_test)
    plot_predictions_vs_actual(y_test, best_pred, best["model"])

    print("\n" + "=" * 55)
    print("  Pipeline complete. All artifacts saved.")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()
