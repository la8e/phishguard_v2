#!/usr/bin/env python3
"""
PhishGuard – Production Predictor (predictor.py)

Boots once, then accepts raw emails in a loop:
  python predictor.py [--vt-key <KEY>]
Type 'exit' or Ctrl-D to quit.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import difflib
import hashlib
import json
import logging
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Project modules ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.preprocess        import production_preprocessing
from src.features.fasttext_features import FastTextFeatureExtractor
from src.features.feature_concat    import FeatureBuilder
from virus_total.vt_client          import VT_Client

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("PhishGuard.Predictor")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR           = ROOT / "models"
LOG_DIR             = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH          = MODEL_DIR / "phishguard_xgb.json"
SCHEMA_PATH         = MODEL_DIR / "feature_schema.json"
SHAP_BACKGROUND_PATH= MODEL_DIR / "shap_background.npy"
WAZUH_JSONL_PATH    = LOG_DIR   / "phishguard_predictions.jsonl"

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SAFE_THRESHOLD     = 0.30
PHISHING_THRESHOLD = 0.70
TARGET_BRANDS      = ["apple", "amazon", "paypal", "microsoft", "google", "netflix", "facebook"]

# Homoglyph → ASCII transliteration table
_HOMOGLYPH_MAP = {
    # Cyrillic look-alikes
    'а':'a','с':'c','е':'e','о':'o','р':'p','х':'x','у':'y',
    'і':'i','ј':'j','ѕ':'s','ԁ':'d','ԛ':'q','ԝ':'w',
    # Greek look-alikes
    'α':'a','ο':'o','ν':'v','ρ':'p','τ':'t','μ':'u',
    # Common digit substitutions
    '0':'o','1':'l','3':'e','5':'s',
}

_IP_RE      = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_FROM_RE    = re.compile(r'^From:\s*.*<(.+?)>', re.M | re.I)
_RP_RE      = re.compile(r'^Return-Path:\s*<(.+?)>',  re.M | re.I)
_HTTP_RE    = re.compile(r'https?://[^\s<>"\'()]+', re.I)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def transliterate_homoglyphs(text: str) -> str:
    """Map visual-trick characters to their ASCII equivalents."""
    text = text.replace('rn', 'm')          # 'rn' → 'm' optical trick
    return ''.join(_HOMOGLYPH_MAP.get(c, c) for c in text)


def _extract_vt_artifacts(raw_text: str):
    """
    Pull URLs, domains, and IPs from raw email text for VirusTotal queries.
    Returns (urls, domains, ips) as lists.
    """
    urls = list(set(_HTTP_RE.findall(raw_text)))
    domains, ips = set(), set()

    # Sender domain
    m = _FROM_RE.search(raw_text)
    if m:
        domains.add(m.group(1).split('@')[-1].strip("<>\"' ").lower())

    for link in urls:
        try:
            netloc = urllib.parse.urlparse(link).netloc.lower().split(':')[0]
            if netloc:
                (ips if _IP_RE.search(netloc) else domains).add(netloc)
        except Exception:
            continue

    return urls, list(domains), list(ips)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class PhishGuardPredictor:
    """
    Central orchestrator.  Loaded once at startup; predict() is called per email.
    """

    def __init__(self, vt_api_key: Optional[str] = None):
        logger.info("Initialising PhishGuard …")
        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_PATH))
        self.builder      = FeatureBuilder()
        self.ft_extractor = FastTextFeatureExtractor()
        self.vt_client    = VT_Client(api_key=vt_api_key)
        self.feature_names= self._load_feature_names()
        self.explainer    = self._init_shap()
        logger.info("PhishGuard ready.")

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _load_feature_names(self) -> List[str]:
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Schema missing: {SCHEMA_PATH}")
        with open(SCHEMA_PATH) as f:
            return json.load(f).get("feature_order", [])

    def _init_shap(self):
        if SHAP_AVAILABLE and SHAP_BACKGROUND_PATH.exists():
            try:
                bg = np.load(SHAP_BACKGROUND_PATH)
                return shap.TreeExplainer(self.model, data=bg)
            except Exception as e:
                logger.error("SHAP init failed: %s", e)
        return None

    # ── Heuristic engine ──────────────────────────────────────────────────────

    def _evaluate_security_heuristics(self, processed: Dict[str, Any]) -> List[Dict]:
        """
        Deterministic rule engine.  Runs independently of the ML model.
        Returns a list of alert dicts (empty if nothing suspicious found).
        """
        alerts: List[Dict] = []
        raw_text    = processed.get("raw_text", "")
        sender_domain = ""

        # Collect all domains to inspect
        domains_to_check = set()
        m = _FROM_RE.search(raw_text)
        if m:
            sender_domain = m.group(1).split('@')[-1].strip("<>\"' ").lower()
            domains_to_check.add(sender_domain)

        for link in _HTTP_RE.findall(raw_text):
            try:
                netloc = urllib.parse.urlparse(link).netloc.lower().split(':')[0]
                if netloc:
                    domains_to_check.add(netloc)
            except Exception:
                continue

        # Per-domain tests
        for raw_domain in domains_to_check:
            try:
                unicode_domain = raw_domain.encode('utf-8').decode('idna')
            except Exception:
                unicode_domain = raw_domain

            core         = unicode_domain.split('.')[0].lower().replace('-', '')
            norm_core    = transliterate_homoglyphs(core)
            has_tricks   = norm_core != core

            for brand in TARGET_BRANDS:
                # Test A: substring impersonation ("paypalsecurity")
                if brand in norm_core and norm_core != brand:
                    alerts.append({
                        "feature":      f"Brand Impersonation ({brand})",
                        "impact_score": 999.8,
                        "actual_value": raw_domain,
                        "direction":    "increases_risk",
                    })
                    break

                # Test B: exact homoglyph match ("аррle" → "apple")
                if has_tricks and norm_core == brand:
                    alerts.append({
                        "feature":      f"Homoglyph Deception ({brand})",
                        "impact_score": 999.7,
                        "actual_value": unicode_domain,
                        "direction":    "increases_risk",
                    })
                    break

                # Test C: Levenshtein typosquatting ("amzaon")
                for part in core.split('-'):
                    sim = difflib.SequenceMatcher(
                        None, brand, transliterate_homoglyphs(part)
                    ).ratio()
                    if 0.80 <= sim < 1.0:
                        alerts.append({
                            "feature":      f"Levenshtein: Brand Spoof ({brand})",
                            "impact_score": 999.8,
                            "actual_value": f"Matched '{part}' ({round(sim*100)}%)",
                            "direction":    "increases_risk",
                        })
                        break

            # Test D: raw IP address as host
            if _IP_RE.search(raw_domain):
                alerts.append({
                    "feature":      "Critical: IP Address as Host",
                    "impact_score": 999.9,
                    "actual_value": raw_domain,
                    "direction":    "increases_risk",
                })

        # Test E: Return-Path spoofing
        rp_m = _RP_RE.search(raw_text)
        if rp_m and sender_domain:
            rp_domain = rp_m.group(1).split('@')[-1].strip("<>\"' ").lower()
            if rp_domain != sender_domain:
                alerts.append({
                    "feature":      "Critical: Sender Spoofing",
                    "impact_score": 999.6,
                    "actual_value": f"From: {sender_domain}  Return-Path: {rp_domain}",
                    "direction":    "increases_risk",
                })

        return alerts

    # ── SHAP explainer ────────────────────────────────────────────────────────

    def explain(self, vector: np.ndarray, decision: str) -> List[Dict]:
        """Compute SHAP values and return human-readable feature contributions."""
        if not self.explainer:
            return []
        try:
            shap_vals = self.explainer.shap_values(vector)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            contributions = []
            for i, val in enumerate(shap_vals[0]):
                name = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
                if name.startswith("emb_") or val == 0:
                    continue
                contributions.append({
                    "feature":      name.replace("_", " ").title(),
                    "impact_score": round(float(val), 4),
                    "direction":    "increases_risk" if val > 0 else "decreases_risk",
                })
            return contributions
        except Exception as e:
            logger.error("SHAP failed: %s", e)
            return []

    # ── Main predict ──────────────────────────────────────────────────────────

    def predict(self, raw_email: str) -> Dict[str, Any]:
        """
        Full pipeline for one email:
          parse → embed → classify → heuristics → SHAP → VT → log → return.
        """
        start_ts = time.time()
        event_id = hashlib.sha256(raw_email.encode()).hexdigest()

        try:
            processed = production_preprocessing(raw_email)
            if not processed:
                return {"event_id": event_id, "status": "rejected"}

            # ML inference
            embedding    = self.ft_extractor.get_embedding(processed.get("clean_text", ""))
            final_vector = self.builder.build_vector(embedding, processed)
            prob         = float(self.model.predict(xgb.DMatrix(final_vector))[0])

            # Threshold
            if prob < SAFE_THRESHOLD:
                decision = "safe"
            elif prob < PHISHING_THRESHOLD:
                decision = "suspicious"
            else:
                decision = "phishing"

            # Heuristics may override ML verdict
            heuristic_alerts = self._evaluate_security_heuristics(processed)
            ai_reasons       = self.explain(final_vector, decision)

            if heuristic_alerts:
                decision = "phishing"
                prob     = 1.0

            # Sort: 999.x heuristic scores always appear above SHAP values
            all_reasons = sorted(
                heuristic_alerts + ai_reasons,
                key=lambda x: abs(x["impact_score"]),
                reverse=True,
            )

            # VirusTotal (parallel, only when API key provided)
            vt_results: Dict = {}
            if self.vt_client._has_key:
                raw_text = processed.get("raw_text", raw_email)
                vt_urls, vt_domains, vt_ips = _extract_vt_artifacts(raw_text)
                if any([vt_urls, vt_domains, vt_ips]):
                    # FIX: corrected indentation — was broken in original
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        fut_urls = executor.submit(
                            self.vt_client.get_reputations, urls=vt_urls
                        )
                        fut_doms = executor.submit(
                            self.vt_client.get_reputations, domains=vt_domains
                        )
                        fut_ips  = executor.submit(
                            self.vt_client.get_reputations, ips=vt_ips
                        )
                        vt_results.update(fut_urls.result())
                        vt_results.update(fut_doms.result())
                        vt_results.update(fut_ips.result())

            # Assemble result
            result: Dict[str, Any] = {
                "event_id":   event_id,
                "probability": round(prob, 4),
                "decision":    decision,
                "top_reasons": all_reasons[:5],
            }
            if vt_results:
                result["reputation_results"] = vt_results
            result["metadata"] = {"runtime_sec": round(time.time() - start_ts, 3)}

            # Append to Wazuh JSONL log
            with open(WAZUH_JSONL_PATH, "a") as f:
                f.write(json.dumps(result) + "\n")

            return result

        except Exception as e:
            logger.error("Pipeline error: %s", e)
            return {"event_id": event_id, "status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# CLI — persistent interactive loop
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhishGuard Predictor")
    parser.add_argument("--vt-key", default=None, help="VirusTotal API key")
    args, _ = parser.parse_known_args()

    print("\n\t[Booting PhishGuard — please wait …]\n")
    predictor = PhishGuardPredictor(vt_api_key=args.vt_key)
    print("\n\t[System ready]\n")

    while True:
        print("\t--- Paste raw email then press Ctrl-D (or type 'exit') ---")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "exit":
                    sys.exit(0)
                lines.append(line)
            except EOFError:
                break

        raw_input = "\n".join(lines)
        if not raw_input.strip():
            continue

        output = predictor.predict(raw_input)
        print("\n\t--- RESULT ---")
        print(json.dumps(output, indent=2, ensure_ascii=False))
