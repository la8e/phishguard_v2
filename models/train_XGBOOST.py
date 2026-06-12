"""
PhishGuard - XGBoost Training Pipeline (train_XGBOOST.py)

- eval metrics saved to models/eval_metrics.json
- confusion matrix PNG -> models/confusion_matrix.png
- PR curve PNG         -> models/pr_curve.png
- ROC curve PNG        -> models/roc_curve.png
- Optuna history PNG   -> models/optuna_history.png
- class distribution   -> printed and stored in metadata
"""
import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from tqdm.auto import tqdm as tqdm_auto
import joblib
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ........................................................................................
# PATHS
ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "embed_output" / "xgboost_features.npz"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH          = MODEL_DIR / "phishguard_xgb.json"
SKL_MODEL_PATH      = MODEL_DIR / "phishguard_xgb.pkl"
METADATA_PATH       = MODEL_DIR / "model_metadata.json"
METRICS_PATH        = MODEL_DIR / "eval_metrics.json"
SHAP_BACKGROUND_PATH= MODEL_DIR / "shap_background.npy"
OPTUNA_DB           = f"sqlite:///{MODEL_DIR / 'optuna_phishguard.db'}"

# ........................................................................................
# HYPERPARAMETERS
RANDOM_STATE      = 42
N_THREADS         = 2
N_TRIALS          = 50
TUNE_SUBSET_SIZE  = 40_000   # rows sampled per Optuna trial
TUNE_BOOST_ROUNDS = 600
TUNE_EARLY_STOP   = 25
FINAL_BOOST_ROUNDS= 2_000
FINAL_EARLY_STOP  = 50
SHAP_BG_SIZE      = 1_024


# ........................................................................................
# PLOT HELPERS
def _save(fig: plt.Figure, path: Path, label: str = "") -> None:
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[plot] {label or path.name} → {path}")

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (r, c), val in np.ndenumerate(cm):
        ax.text(c, r, f"{val:,}", ha="center", va="center",
                color="white" if val > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Phish"])
    ax.set_yticklabels(["Legit", "Phish"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    _save(fig, path, "confusion matrix")

def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#e91e63", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path, "PR curve")

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1565c0", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path, "ROC curve")

def plot_optuna_history(study: optuna.Study, path: Path) -> None:
    values = [t.value for t in study.trials if t.value is not None]
    best   = np.maximum.accumulate(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(values, alpha=0.5, label="Trial AUC-PR", color="#78909c")
    ax.plot(best,   lw=2,      label="Best so far",  color="#00897b")
    ax.set_xlabel("Trial"); ax.set_ylabel("AUC-PR")
    ax.set_title("Optuna Optimisation History")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path, "Optuna history")

# ........................................................................................
# MAIN
def main() -> None:
    # 1. Load data
    print("Loading dataset ...")
    npz = np.load(DATA_PATH, mmap_mode="r")
    X   = np.array(npz["X"], dtype=np.float32)
    y   = np.array(npz["y"], dtype=np.int32)

    # Drop unlabelled rows (label == -1)
    mask = y != -1
    X, y = X[mask], y[mask]
    print(f"Dataset shape after filtering: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # 2. Class imbalance
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos
    print(f"Train positives: {pos:,}  negatives: {neg:,}  scale_pos_weight: {scale_pos_weight:.4f}")
    # 3. Base params
    BASE_PARAMS = {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",
        "tree_method":      "hist",
        "grow_policy":      "lossguide",
        "verbosity":        0,
        "nthread":          N_THREADS,
        "max_bin":          128,
        "scale_pos_weight": scale_pos_weight,
    }
    # 4. Optuna tuning
    def objective(trial: optuna.Trial) -> float:
        rng = np.random.RandomState(RANDOM_STATE + trial.number)
        idx = rng.choice(X_train.shape[0], min(TUNE_SUBSET_SIZE, X_train.shape[0]), replace=False)
        dtrain = xgb.DMatrix(X_train[idx], label=y_train[idx], nthread=N_THREADS)

        params = {**BASE_PARAMS,
            "learning_rate":    trial.suggest_float("learning_rate",    0.01,  0.15,  log=True),
            "max_depth":        trial.suggest_int(  "max_depth",        4,     10),
            "subsample":        trial.suggest_float("subsample",        0.6,   1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6,   1.0),
            "min_child_weight": trial.suggest_int(  "min_child_weight", 1,     15),
            "gamma":            trial.suggest_float("gamma",            0.0,   5.0),
            "lambda":           trial.suggest_float("lambda",           1e-3,  10.0, log=True),
            "alpha":            trial.suggest_float("alpha",            1e-3,  10.0, log=True),
        }
        cv = xgb.cv(
            params=params, dtrain=dtrain,
            num_boost_round=TUNE_BOOST_ROUNDS,
            nfold=3, stratified=True,
            early_stopping_rounds=TUNE_EARLY_STOP,
            seed=RANDOM_STATE, verbose_eval=False,
        )
        score = float(cv["test-aucpr-mean"].max())
        del dtrain, cv
        gc.collect()
        return score

    study = optuna.create_study(
        study_name="phishguard_tune",
        storage=OPTUNA_DB,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    print(f"Starting Optuna ({N_TRIALS} trials) ...")
    with tqdm_auto(total=N_TRIALS, desc="Optuna trials", unit="trial") as pbar:
        def _callback(study, trial):
            pbar.set_postfix_str(f"best={study.best_value:.4f}")
            pbar.update(1)
        study.optimize(
            objective, n_trials=N_TRIALS, n_jobs=1,
            timeout=60 * 60 * 4, gc_after_trial=True,
            callbacks=[_callback],
        )

    best_params = study.best_trial.params
    print("Best hyperparameters:", best_params)
    plot_optuna_history(study, MODEL_DIR / "optuna_history.png")
    # 5. Final model training
    final_params = {**BASE_PARAMS, **best_params, "scale_pos_weight": scale_pos_weight}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_test,  label=y_test)

    print("Training final model ...")
    model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=FINAL_BOOST_ROUNDS,
        evals=[(dval, "val")],
        early_stopping_rounds=FINAL_EARLY_STOP,
        verbose_eval=100,
    )
    model.save_model(MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    # 6. Evaluation
    print("Evaluating ...")
    y_prob   = model.predict(xgb.DMatrix(X_test))
    # Find best F1 threshold on PR curve
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1s        = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx   = int(np.argmax(f1s))
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    y_binary = (y_prob > best_thresh).astype(int)
    metrics = {
        "accuracy":        round(float(accuracy_score(y_test, y_binary)),             4),
        "precision":       round(float(precision_score(y_test, y_binary)),            4),
        "recall":          round(float(recall_score(y_test, y_binary)),               4),
        "f1_score":        round(float(f1_score(y_test, y_binary)),                   4),
        "roc_auc":         round(float(roc_auc_score(y_test, y_prob)),                4),
        "average_precision": round(float(average_precision_score(y_test, y_prob)),    4),
        "best_threshold":  round(best_thresh,                                         4),
        "best_f1":         round(float(f1s[best_idx]),                                4),
        "train_samples":   int(len(y_train)),
        "test_samples":    int(len(y_test)),
    }
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    # save metrics + plots
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved → {METRICS_PATH}")

    plot_confusion_matrix(y_test, y_binary,  MODEL_DIR / "confusion_matrix.png")
    plot_pr_curve(y_test, y_prob,            MODEL_DIR / "pr_curve.png")
    plot_roc_curve(y_test, y_prob,           MODEL_DIR / "roc_curve.png")
    # 7. SHAP background sample
    rng = np.random.RandomState(RANDOM_STATE)
    bg  = X[rng.choice(X.shape[0], min(SHAP_BG_SIZE, X.shape[0]), replace=False)]
    np.save(SHAP_BACKGROUND_PATH, bg)
    print(f"SHAP background saved → {SHAP_BACKGROUND_PATH}")
    # 8. Metadata
    metadata = {
        "samples":           int(len(y)),
        "features":          int(X.shape[1]),
        "train_samples":     metrics["train_samples"],
        "test_samples":      metrics["test_samples"],
        "best_params":       best_params,
        "scale_pos_weight":  float(scale_pos_weight),
        "class_distribution":{"phish": pos, "legit": neg},
        "timestamp":         time.time(),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved → {METADATA_PATH}")
    # 9. sklearn wrapper (for tools that expect sklearn API)
    clf = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=N_THREADS,
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf, SKL_MODEL_PATH)
    print(f"sklearn wrapper saved → {SKL_MODEL_PATH}")
    print("\nTraining pipeline complete.")

if __name__ == "__main__":
    main()
