"""
PhishGuard: FastText Feature Extractor (fasttext_features.py)
Loads the pre-trained cc.en.300.bin model once and exposes:
  • get_embedding(text)          - single-string production inference
  • generate_training_data(...)  - batch .npy export for the training pipeline
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models.fasttext import load_facebook_vectors
from tqdm import tqdm

# Paths
HERE          = Path(__file__).resolve().parent
PROJECT_ROOT  = HERE.parent.parent
FASTTEXT_PATH = Path(os.environ.get("FASTTEXT_MODEL_PATH", str(HERE / "fastText/cc.en.300.bin")))
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "phishguard_features.csv"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "processed" / "embed_output"
EDA_DIR       = PROJECT_ROOT / "data" / "processed" / "eda"

class FastTextFeatureExtractor:
    """
    Wrapper around the pre-trained Facebook FastText model.
    Model is loaded once into RAM at construction time.
    """
    def __init__(self, model_path: Path = FASTTEXT_PATH):
        print(f"[FastText] Loading model from {model_path} ...")
        self.ft_model  = load_facebook_vectors(str(model_path))
        self.embed_dim = self.ft_model.vector_size
        print(f"[FastText] Ready - embedding dim: {self.embed_dim}")

    # Production ........................................................................................
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Return the mean FastText vector for all recognised tokens in *text*.
        Falls back to a zero vector for empty / OOV-only inputs.
        """
        if not text or not isinstance(text, str):
            return np.zeros(self.embed_dim, dtype=np.float32)
        vectors = [self.ft_model[w] for w in text.split() if w in self.ft_model]
        if not vectors:
            return np.zeros(self.embed_dim, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    # Training batch export 
    def generate_training_data(
        self,
        features_csv: Path = FEATURES_PATH,
        output_dir:   Path = OUTPUT_DIR,
        eda_dir:      Path = EDA_DIR,
    ) -> None:
        """
        Read the preprocessed CSV, embed every clean_text row, and save:
          embed_output/X_embeddings.npy  - (N, 300) float32 embedding matrix
          embed_output/y_labels.npy      - (N,) int32 label vector
          embed_output/embedding_stats.json - basic embedding stats
          eda/embedding_norm_dist.png    - L2-norm histogram
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(eda_dir).mkdir(parents=True, exist_ok=True)

        print(f"[FastText] Reading {features_csv} ...")
        df     = pd.read_csv(features_csv)
        texts  = df["clean_text"].fillna("").astype(str).values
        labels = df["label"].astype(np.int32).to_numpy()
        print(f"[FastText] {len(texts):,} samples loaded.")

        # Embed ........................................................................................
        embeddings = np.zeros((len(texts), self.embed_dim), dtype=np.float32)
        for i in tqdm(range(len(texts)), desc="Embedding", unit="email"):
            embeddings[i] = self.get_embedding(texts[i])

        # Save primary artifacts
        x_path = Path(output_dir) / "X_embeddings.npy"
        y_path = Path(output_dir) / "y_labels.npy"
        np.save(x_path, embeddings)
        np.save(y_path, labels)
        print(f"[FastText] Saved:\n  -> {x_path}\n  -> {y_path}")

        # embedding statistics JSON
        norms = np.linalg.norm(embeddings, axis=1)
        zero_mask = (norms == 0)
        stats = {
            "total_samples":    int(len(embeddings)),
            "embedding_dim":    int(self.embed_dim),
            "zero_vectors":     int(zero_mask.sum()),       # OOV / empty rows
            "mean_l2_norm":     float(norms[~zero_mask].mean()) if (~zero_mask).any() else 0.0,
            "std_l2_norm":      float(norms[~zero_mask].std())  if (~zero_mask).any() else 0.0,
            "min_l2_norm":      float(norms.min()),
            "max_l2_norm":      float(norms.max()),
        }
        stats_path = Path(output_dir) / "embedding_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"[FastText] Embedding stats -> {stats_path}")

        # L2-norm histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(norms[~zero_mask], bins=60, edgecolor="black", color="#3f51b5", alpha=0.85)
        ax.set_title("FastText Embedding L2-Norm Distribution (non-zero vectors)")
        ax.set_xlabel("L2 Norm")
        ax.set_ylabel("Count")
        plot_path = Path(eda_dir) / "embedding_norm_dist.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"[FastText] Norm plot -> {plot_path}")

# ........................................................................................
# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhishGuard FastText Feature Extractor")
    parser.add_argument(
        "mode", choices=["train", "prod"],
        help="train: generate .npy files  |  prod: quick self-test",
    )
    parser.add_argument("--csv",   default=str(FEATURES_PATH), help="Input CSV (train mode)")
    parser.add_argument("--out",   default=str(OUTPUT_DIR),    help="Output directory (train mode)")
    parser.add_argument("--model", default=str(FASTTEXT_PATH), help="Path to cc.en.300.bin")
    args = parser.parse_args()

    extractor = FastTextFeatureExtractor(model_path=Path(args.model))
    if args.mode == "train":
        print("=== TRAINING MODE ===")
        extractor.generate_training_data(
            features_csv=Path(args.csv),
            output_dir=Path(args.out),
        )
    else:
        print("=== PRODUCTION SELF-TEST ===")
        test = "urgent password reset click link <URL>"
        vec  = extractor.get_embedding(test)
        print(f"Input : '{test}'")
        print(f"Shape : {vec.shape}")
        print(f"Preview: {vec[:5]} ...")
