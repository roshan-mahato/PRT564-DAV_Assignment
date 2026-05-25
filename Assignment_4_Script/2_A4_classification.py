"""
Assessment 4 classification pipeline for multimorbidity risk grouping.

The script trains three taught classifiers, evaluates them with cross-validation,
and writes outputs under outputs/Assignment_4/classification.

Run from the repo root:
    python Assignment_4_Script/2_A4_classification.py
"""

import warnings
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
DATA = PROJECT_ROOT / "data" / "regression_dataset" / "regression_dataset.csv"
OUT  = PROJECT_ROOT / "outputs" / "Assignment_4" / "classification"
OUT.mkdir(parents=True, exist_ok=True)

def ensure_dataset():
    if not DATA.exists():
        print("Dataset missing — attempting to run shared Assignment_2_Script/2_regression_dataset_from_transformed.py")
        script = PROJECT_ROOT / "Assignment_2_Script" / "2_regression_dataset_from_transformed.py"
        if script.exists():
            subprocess.run([sys.executable, str(script)], check=True)
        else:
            raise FileNotFoundError("Dataset missing and local preprocessing script not found.")

def build_target(df):
    COMORB_RATE_COLS = [c for c in df.columns if "nhs_comorbidity_all_condition" in c and c.endswith("_rate")]
    df = df.copy()
    df["disease_type_count"] = (df[COMORB_RATE_COLS] > 0.05).sum(axis=1)
    t33 = df["disease_type_count"].quantile(0.33)
    t66 = df["disease_type_count"].quantile(0.66)
    if t33 == t66:
        t33 -= 1
    bins = sorted(set([-1, t33, t66, 100]))
    labels = ["Low","Medium","High"][: len(bins)-1]
    df["multimorbidity_class"] = pd.cut(df["disease_type_count"], bins=bins, labels=labels)
    return df, t33, t66

def select_features(df):
    all_rate_cols = [c for c in df.columns if c.endswith("_rate")]
    LEAKAGE_PREFIXES = ["nhs_comorbidity_all_condition","nhs_abs_condition","nhs_comorbidity_no_condition"]
    drop_leakage = [c for c in all_rate_cols if any(p in c for p in LEAKAGE_PREFIXES)]
    feature_cols = ["age"] + [c for c in all_rate_cols if c not in drop_leakage]

    X_raw = df[feature_cols].copy()
    vt = VarianceThreshold(threshold=1e-4)
    vt.fit(X_raw)
    kept = [c for c,k in zip(feature_cols, vt.get_support()) if k]
    X_raw = X_raw[kept]

    # Correlation pruning
    corr_abs = X_raw.corr().abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    drop_corr = {col for col in upper.columns if any(upper[col] > 0.95)}
    features = [c for c in X_raw.columns if c not in drop_corr]
    X_final = X_raw[features]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_final), columns=features, index=X_final.index)
    return X_scaled, features

def run_models(X, y):
    CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results = {}
    # Naive Bayes
    # nb = GaussianNB()
    nb_pipeline = Pipeline([('scaler', StandardScaler()), ('model', GaussianNB())])
    nb_preds = cross_val_predict(nb_pipeline, X, y, cv=CV)
    results["Naive Bayes"] = {"model": nb_pipeline, "preds": nb_preds}

    # SVM with RBF kernel
    svm_grid = {"C": [0.1, 1.0, 5.0], "gamma": ["scale", "auto"]}
    svm_model = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    svm_gs = GridSearchCV(SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
                           param_grid=svm_grid, cv=CV, scoring="f1_macro", n_jobs=-1)
    svm_pipeline = Pipeline([('scaler', StandardScaler()), ('gs', svm_gs)])
    svm_pipeline.fit(X, y)
    best_svm = svm_pipeline.named_steps['gs'].best_estimator_
    svm_preds = cross_val_predict(Pipeline([('scaler', StandardScaler()), ('model', best_svm)]), X, y, cv=CV)
    results["SVM"] = {"model": best_svm, "preds": svm_preds}

    # Random Forest
    rf_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [6, None],
        "rf__min_samples_leaf": [1, 3],
    }

    rf_base_pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('rf', RandomForestClassifier(class_weight="balanced", random_state=42))
    ])
    rf_gs = GridSearchCV(rf_base_pipe, param_grid=rf_grid, cv=CV, scoring="f1_macro", n_jobs=-1)
    rf_gs.fit(X, y)
    best_rf_pipe = rf_gs.best_estimator_
    rf_preds = cross_val_predict(best_rf_pipe, X, y, cv=CV)
    clean_params = {k.replace("rf__", ""): v for k, v in rf_gs.best_params_.items()}
    
    results["Random Forest"] = {
        "model": best_rf_pipe.named_steps['rf'], 
        "preds": rf_preds, 
        "best_params": clean_params, 
        "best_score": rf_gs.best_score_
    }

    return results

def evaluate(results, y, X, features, t33, t66):
    model_names = list(results.keys())
    metrics_rows = []
    for name in model_names:
        preds = results[name]["preds"]
        acc = accuracy_score(y, preds)
        f1m = f1_score(y, preds, average="macro")
        f1w = f1_score(y, preds, average="weighted")
        metrics_rows.append({"Model": name, "Accuracy": round(acc,3), "Macro F1": round(f1m,3), "Weighted F1": round(f1w,3)})
        print(f"\n{name}")
        print(classification_report(y, preds))

    pd.DataFrame(metrics_rows).to_csv(OUT / "model_metrics_medium.csv", index=False)

    fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names),4))
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(y, results[name]["preds"], labels=["Low","Medium","High"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low","Medium","High"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(OUT / "eval_confusion_matrices_medium.png", dpi=150)
    plt.close()

    y_bin = label_binarize(y, classes=["Low","Medium","High"]) if len(set(y))>1 else None
    if y_bin is not None:
        fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names),4), sharey=True)
        if len(model_names) == 1:
            axes = [axes]
        for ax, name in zip(axes, model_names):
            model = results[name]["model"]
            try:
               
                pipe = Pipeline([('scaler', StandardScaler()), ('cls', model)])

                y_prob =cross_val_predict(pipe, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), method="predict_proba")
                auc_scores = []
                for i, cls in enumerate(["Low","Medium","High"]):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    auc_val = roc_auc_score(y_bin[:, i], y_prob[:, i])
                    auc_scores.append(auc_val)
                    ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.2f})")
                macro_auc = np.mean(auc_scores)
                ax.plot([0,1],[0,1],"k--",linewidth=0.8)
                ax.set_title(f"{name}\nMacro AUC={macro_auc:.2f}")
                ax.set_xlabel("False positive rate")
                ax.legend(fontsize=8)
            except Exception as e:
                ax.text(0.5,0.5,f"ROC not available\n{e}",ha="center")
                ax.set_title(name)
        plt.tight_layout()
        plt.savefig(OUT / "eval_roc_auc_medium.png", dpi=150)
        plt.close()

    if "Random Forest" in results:
        rf = results["Random Forest"]["model"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf.fit(X_scaled, y)
        importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
        importances.to_csv(OUT / "rf_feature_importance_medium.csv", header=["importance"])

    errors = {name: (np.array(y) != np.array(results[name]["preds"])).astype(int) for name in model_names}
    from itertools import combinations
    rows = []
    
    for (m1, e1), (m2, e2) in combinations(errors.items(), 2):
        t_stat, t_pval = stats.ttest_rel(e1, e2)
        w_stat, w_pval = stats.wilcoxon(e1, e2, zero_method="wilcox")
        concl = "Significant difference" if t_pval < 0.05 else "No significant difference"
        
        rows.append({
            "Model Comparison": f"{m1} vs {m2}",
            "t-statistic": round(t_stat, 4),
            "t-test p-value": round(t_pval, 4),
            "Wilcoxon p-value": round(w_pval, 4),
            "Hypothesis Conclusion (alpha=0.05)": concl
        })
        
    pd.DataFrame(rows).to_csv(OUT / "statistical_tests_assignment4.csv", index=False)

    print("Assessment 4 evaluation outputs saved to:", OUT)

def main():
    ensure_dataset()
    df = pd.read_csv(DATA)
    df, t33, t66 = build_target(df)
    X, features = select_features(df)
    y = df["multimorbidity_class"].astype(str)
    results = run_models(X, y)
    evaluate(results, y, X, features, t33, t66)

if __name__ == "__main__":
    main()
