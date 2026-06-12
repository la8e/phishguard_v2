"""
PhishGuard: Feature Concatenation Pipeline (feature_concat.py)

Training mode  (--mode train):
  Align embeddings with the CSV, compute manual security features,
  fit a StandardScaler, concatenate, and save everything.

Production mode (imported):
  FeatureBuilder loads schema + scaler once and produces the
  314-dimensional model-input vector on demand.
"""
import argparse, json, logging
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feature_concat")

# ............................................................................
# PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed" / "embed_output"
MODEL_DIR    = PROJECT_ROOT / "models"
EDA_DIR      = PROJECT_ROOT / "data" / "processed" / "eda"

CSV_PATH    = DATA_DIR.parent / "phishguard_features.csv"
EMBED_PATH  = DATA_DIR / "X_embeddings.npy"
LABEL_PATH  = DATA_DIR / "y_labels.npy"
OUTPUT_PATH = DATA_DIR / "xgboost_features.npz"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
SCALER_PATH = MODEL_DIR / "manual_scaler.pkl"

# ............................................................................
# FEATURE DEFINITIONS (order is fixed - do not reorder)
NUMERIC_BASE = [
    "urgent_words_count", "digit_ratio", "body_entropy",
    "html_present", "auth_headers_present",
    "spf_result", "dkim_result", "dmarc_result", "received_count",
]
COUNT_FEATURES = ["urls_count", "domains_count", "ip_urls_count", "attachment_names_count",]
HEADER_FEATURES = ["return_path_mismatch"]
MANUAL_FEATURES = NUMERIC_BASE + COUNT_FEATURES + HEADER_FEATURES  # 14 total

# ............................................................................
# UTILITY (used in both training and production)
def extract_domain(email_string: str) -> str:
    """Parse the domain from an RFC-5321 address string; returns '' on failure."""
    if not isinstance(email_string, str) or not email_string:
        return ""
    _, addr = parseaddr(email_string)
    return addr.split("@")[-1].lower() if "@" in addr else ""

# ............................................................................
# TRAINING:  manual feature extraction from DataFrame
def extract_manual_features(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorised extraction of all 14 manual security features from a DataFrame.
    Missing columns are zeroed rather than raising.
    Returns a float32 array of shape (N, 14).
    """
    # Ensure expected columns exist with numeric types & correct defaults
    _FILL_DEFAULTS = {
        "urgent_words_count":     0,
        "digit_ratio":            0.0,
        "body_entropy":           0.0,
        "html_present":           0,
        "auth_headers_present":   0,
        "spf_result":            -1, # -1 = absent, not fail
        "dkim_result":           -1,
        "dmarc_result":          -1,
        "received_count":         0,
        "urls_count":             0,
        "domains_count":          0,
        "ip_urls_count":          0,
        "attachment_names_count": 0
    }
    for col in NUMERIC_BASE + COUNT_FEATURES:
        if col not in df.columns:
            df[col] = _FILL_DEFAULTS.get(col, 0)
    df[NUMERIC_BASE + COUNT_FEATURES] = (
        df[NUMERIC_BASE + COUNT_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
            .fillna(pd.Series(_FILL_DEFAULTS))
    )

    # Vectorised return-path mismatch (avoids a slow row-by-row apply)
    sender_col   = df.get("from_header", df.get("sender", pd.Series([""] * len(df))))
    sender_doms  = sender_col.apply(extract_domain)
    return_doms  = df.get("return_path", pd.Series([""] * len(df))).apply(extract_domain)
    df["return_path_mismatch"] = (
        (sender_doms != "") & (return_doms != "") & (sender_doms != return_doms)
    ).astype(int)
    return df[MANUAL_FEATURES].values.astype(np.float32)

# ............................................................................
# EDA - feature-matrix visualisations
def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved plot → %s", path)

def save_feature_eda(X_manual: np.ndarray, y: np.ndarray, eda_dir: str) -> None:
    """
    Persist feature-level EDA artifacts:
      eda/feature_correlation.png    - heatmap of pairwise Pearson r between manual features
      eda/feature_boxplots.png       - per-feature box-plots split by label (0 vs 1)
      eda/feature_means_by_label.csv - mean value per feature per label class
    """
    eda_path = Path(eda_dir)
    eda_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X_manual, columns=MANUAL_FEATURES)
    df["label"] = y
    # 1. Correlation heatmap 
    corr = df[MANUAL_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(MANUAL_FEATURES)))
    ax.set_yticks(range(len(MANUAL_FEATURES)))
    ax.set_xticklabels(MANUAL_FEATURES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(MANUAL_FEATURES, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Manual Feature Correlation (Pearson)")
    _save(fig, eda_path / "feature_correlation.png")
    # 2. Box-plots by label 
    labeled = df[df["label"].isin([0, 1])]
    n = len(MANUAL_FEATURES)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for i, feat in enumerate(MANUAL_FEATURES):
        data0 = labeled.loc[labeled["label"] == 0, feat]
        data1 = labeled.loc[labeled["label"] == 1, feat]
        axes[i].boxplot([data0, data1], labels=["Legit", "Phish"],
                        patch_artist=True,
                        boxprops=dict(facecolor="#b3e5fc"))
        axes[i].set_title(feat, fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions by Label", y=1.01)
    fig.tight_layout()
    _save(fig, eda_path / "feature_boxplots.png")
    # 3. Mean value per feature per label 
    means = labeled.groupby("label")[MANUAL_FEATURES].mean()
    means.to_csv(eda_path / "feature_means_by_label.csv")
    logger.info("Feature means by label saved.")

# ............................................................................
# TRAINING PIPELINE
def train_pipeline() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True,  exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading CSV …")
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

    logger.info("Loading embeddings and labels …")
    embeddings = np.load(EMBED_PATH)
    y          = np.load(LABEL_PATH).astype(int)

    # Align all three sources to the shortest length
    n = min(len(df), len(embeddings), len(y))
    df, embeddings, y = df.iloc[:n], embeddings[:n], y[:n]

    # Drop unlabelled rows
    mask       = y != -1
    df         = df.loc[mask].reset_index(drop=True)
    embeddings = embeddings[mask]
    y          = y[mask]
    logger.info("Labelled samples after filtering: %d", len(y))
    if len(y) == 0:
        raise ValueError("No labelled samples found - cannot train.")

    # Label distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info("Label distribution: %s", dict(zip(unique.tolist(), counts.tolist())))
    logger.info("Extracting manual features …")
    X_manual = extract_manual_features(df)

    # feature EDA ------------------------------
    save_feature_eda(X_manual, y, str(EDA_DIR))

    logger.info("Fitting StandardScaler …")
    scaler   = StandardScaler()
    X_manual = scaler.fit_transform(X_manual)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)

    logger.info("Concatenating embeddings + manual features …")
    X = np.hstack([embeddings.astype(np.float32), X_manual]).astype(np.float32)

    np.savez_compressed(OUTPUT_PATH, X=X, y=y)
    logger.info("Feature matrix saved → %s  shape=%s", OUTPUT_PATH, X.shape)
    # Save schema (feature names + ordering)
    schema = {
        "embedding_dim":   int(embeddings.shape[1]),
        "manual_features": MANUAL_FEATURES,
        "feature_order":   [f"emb_{i}" for i in range(embeddings.shape[1])] + MANUAL_FEATURES,
    }
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=4)
    logger.info("Schema saved → %s", SCHEMA_PATH)
    logger.info("Training pipeline complete. Final shape: %s", X.shape)

# ............................................................................
# PRODUCTION - FeatureBuilder class
class FeatureBuilder:
    """
    Loads schema and scaler once at startup.
    Call build_vector(embedding, manual_dict) per email during inference.
    """
    def __init__(self):
        if not SCHEMA_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Schema or scaler not found in {MODEL_DIR}. Run --mode train first."
            )
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        self.embedding_dim   = schema["embedding_dim"]
        self.manual_features = schema["manual_features"]
        self.scaler          = joblib.load(SCALER_PATH)
        logger.info("FeatureBuilder ready (embedding_dim=%d).", self.embedding_dim)

    def build_vector(self, embedding_vector: np.ndarray, manual_dict: Dict[str, Any]) -> np.ndarray:
        """
        Combine a FastText embedding with the manual feature dict into the
        final 314-dim input vector expected by XGBoost.
        Steps:
          1. Compute return_path_mismatch from from_header / return_path.
          2. Extract features in schema order.
          3. Scale with the fitted StandardScaler.
          4. Hstack [embedding | scaled_manual] and return as float32.
        """
        if len(embedding_vector) != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, got {len(embedding_vector)}"
            )
        # Recompute mismatch at inference time (not stored in the dict reliably)
        sender   = manual_dict.get("from_header", manual_dict.get("sender", ""))
        ret_path = manual_dict.get("return_path", "")
        manual_dict["return_path_mismatch"] = int(
            bool(extract_domain(sender) and extract_domain(ret_path)
                 and extract_domain(sender) != extract_domain(ret_path))
        )

        manual_values = np.array(
            [float(manual_dict.get(f, 0.0)) for f in self.manual_features],
            dtype=np.float32,
        ).reshape(1, -1)

        scaled = self.scaler.transform(manual_values)
        return np.hstack([embedding_vector.reshape(1, -1), scaled]).astype(np.float32)

# ............................................................................
# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhishGuard Feature Pipeline")
    parser.add_argument(
        "--mode", choices=["train", "prod"], required=True,
        help="train: build scaler/schema  |  prod: run self-test",
    )
    args = parser.parse_args()
    if args.mode == "train":
        train_pipeline()
    else:
        logger.info("Production self-test …")
        builder = FeatureBuilder()
        fake_emb = np.random.rand(builder.embedding_dim).astype(np.float32)
        fake_manual: Dict[str, Any] = {
            "urgent_words_count": 3, "digit_ratio": 0.12, "body_entropy": 4.5,
            "html_present": 1, "auth_headers_present": 0, "spf_result": -1,
            "dkim_result": -1, "dmarc_result": -1, "received_count": 8,
            "urls_count": 5, "domains_count": 3, "ip_urls_count": 1,
            "attachment_names_count": 0,
            "from_header": "security@evil.com", "return_path": "noreply@bank.com",
        }
        vec = builder.build_vector(fake_emb, fake_manual)
        print(f"\nSUCCESS - output shape: {vec.shape}")
