"""
Self-contained medium EDA for Assignment 4 (copy adapted for update_asign_4).
This folder contains a small preprocessing script to build the regression
dataset if needed. Outputs are written under `output/eda` in this folder.

Run from the repo or by changing into this folder:
    python assignment4_eda_medium.py
"""

from logging import root
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA = PROJECT_ROOT / "data" / "regression_dataset" / "regression_dataset.csv"
OUT  = PROJECT_ROOT / "outputs" / "Assignment_4" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

def ensure_dataset():
    if not DATA.exists():
        print("regression_dataset.csv not found — attempting to build it using Assignment_2_Script/2_regression_dataset_from_transformed.py")
        script = PROJECT_ROOT / "Assignment_2_Script" / "2_regression_dataset_from_transformed.py"
        if script.exists():
            subprocess.run([sys.executable, str(script)], check=True)
        else:
            raise FileNotFoundError("Could not find dataset or the preprocessing script (Assignment_2_Script/2_regression_dataset_from_transformed.py)")

def make_target(df):
    COMORB_RATE_COLS = [c for c in df.columns if "nhs_comorbidity_all_condition" in c and c.endswith("_rate")]
    df = df.copy()
    df["disease_type_count"] = (df[COMORB_RATE_COLS] > 0.05).sum(axis=1)
    t33 = df["disease_type_count"].quantile(0.33)
    t66 = df["disease_type_count"].quantile(0.66)
    if t33 == t66:
        t33 -= 1
    bins = sorted(set([-1, t33, t66, 100]))
    labels = ["Low","Medium","High"][: len(bins) - 1]
    df["multimorbidity_class"] = pd.cut(df["disease_type_count"], bins=bins, labels=labels)
    return df, t33, t66

def main():
    ensure_dataset()
    df = pd.read_csv(DATA)
    df, t33, t66 = make_target(df)

    # Class distribution
    counts = df["multimorbidity_class"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0","#DD8452","#C44E52"])
    ax.set_title("Class distribution — multimorbidity burden")
    ax.set_ylabel("Number of age groups")
    try:
        fig.tight_layout()
    except Exception:
        pass
    plt.savefig(OUT / "eda_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Key feature boxplots
    KEY_FEATURES = [
        "age",
        "nhs_seifa__decile_1_lowest_rate",
        "nhs_activity__did_not_meet_2014_physical_activity_guidelines_rate",
        "nhs_smoking__current_daily_smoker_rate",
        "nhs_bmi__obese_class_1_rate",
    ]
    available = [f for f in KEY_FEATURES if f in df.columns]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(4*len(available),4))
        if len(available) == 1:
            axes = [axes]
        for ax, feat in zip(axes, available):
            data_by_class = [df.loc[df["multimorbidity_class"] == cls, feat].dropna().values for cls in ["Low","Medium","High"]]
            bp = ax.boxplot(data_by_class, patch_artist=True, tick_labels=["Low","Medium","High"])
            for patch, color in zip(bp["boxes"],["#4C72B0","#DD8452","#C44E52"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(feat.replace("nhs_",""))
        try:
            fig.tight_layout()
        except Exception:
            pass
        plt.savefig(OUT / "eda_key_feature_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Top correlations
    rate_cols = [c for c in df.columns if c.endswith("_rate")]
    if rate_cols:
        corr = df[rate_cols].corrwith(df["disease_type_count"]).abs().sort_values(ascending=False)
        top = corr.head(25)
        top.to_csv(OUT / "eda_top25_correlations.csv")
        top_cols = top.index.tolist()
        corrmat = df[top_cols].corr()
        # drop any columns with zero variance to avoid divide-by-zero in correlation
        valid_top_cols = [c for c in top_cols if df[c].std(skipna=True) != 0]
        if len(valid_top_cols) >= 2:
            corrmat = df[valid_top_cols].corr()
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corrmat, cmap="vlag", center=0, ax=ax)
            ax.set_title("Correlation matrix — top 25 features")
            try:
                fig.tight_layout()
            except Exception:
                pass
            plt.savefig(OUT / "eda_top25_correlation_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()
        plt.close()

    summary = {"rows": len(df), "cols": len(df.columns), "t33": float(t33), "t66": float(t66)}
    pd.DataFrame([summary]).to_csv(OUT / "eda_summary_medium.csv", index=False)
    print("EDA (medium) outputs saved to:", OUT)

if __name__ == "__main__":
    main()
