# PhishGuard

A hybrid phishing email detection system combining FastText semantic embeddings,
XGBoost classification, deterministic security heuristics, SHAP explainability,
and VirusTotal threat intelligence.

## Requirements
- Python 3.8 or higher
- Git
- wget (Linux/macOS - pre-installed on most systems)
- ~5 GB free disk space (for the FastText model)
- A VirusTotal API key (free tier at virustotal.com) - optional

## Installation
Run the setup script. It will clone the repository, create a virtual environment,
install all dependencies, and download the FastText model (~4.2 GB).

**Linux / macOS:**
```bash
bash setup.sh
```
**Windows:**
```bash
python setup.py
```

This takes 10-20 minutes depending on your connection. Run it once.

## Running
Activate the virtual environment, then launch the predictor.

```bash
cd phishguard_v2
source .venv/bin/activate
python src/predictor.py
```

With VirusTotal enrichment:

```bash
python src/predictor.py --vt-key YOUR_API_KEY
```

The system will load all models into memory (this takes a few minutes for FastText),
then enter an interactive loop. Paste a raw email, press Enter twice, then CTRL+D or type
`EOF` on a new line to submit.

Results are printed as JSON and appended to `logs/phishguard_predictions.jsonl`.

## Output Example
```json
{
  "event_id": "a3f8b2c1...",
  "probability": 1.0,
  "decision": "phishing",
  "top_reasons": [
    { "feature": "Brand Impersonation (paypal)",
      "impact_score": 999.8,
      "direction": "increases_risk" }
  ],
  "metadata": { "runtime_sec": 0.217 }
}
```

## Project Structure
```md
phishguard_v2/
├─ src/
│  ├─ predictor.py             # Main entry point
│  └─ features/
│     ├─ preprocess.py         # Email parsing and feature extraction
│     ├─ fasttext_features.py  # Embedding generation
│     └─ feature_concat.py     # Feature pipeline and scaler
├─ models/                     # Trained model artifacts (XGBoost, scaler, schema)
├─ logs/                       # JSONL prediction log (Wazuh-compatible)
├─ requirements.txt
├─ setup.sh                    # linux
└─ setup.py                    # win
```
