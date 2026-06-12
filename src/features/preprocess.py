"""
PhishGuard: Preprocessing Engine (preprocess.py)
Handles: raw email ingestion (EML + CSV), English filtering,
feature extraction, deduplication, artifact saving, and
production single-email preprocessing.
"""
import os, re, sys, csv, math, time, email, html, hashlib, logging
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, Iterable, List, Optional
# 3rdP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from email.utils import parseaddr
from urllib.parse import urlparse

# Increase CSV field limit before pandas import
_max_int = sys.maxsize
# maximum integer on the platform. If setting it directly causes OverflowError, the limit is reduced repeatedly until it works.
while True:
    try:
        csv.field_size_limit(_max_int)
        break
    except OverflowError:
        _max_int = int(_max_int / 10)

# Logging
logging.getLogger("urlextract").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocess")

# ........................................................................................
# CONFIG
_HERE               = Path(__file__).resolve().parent       # src/features
_PROJECT_ROOT       = _HERE.parent.parent                   # project root
RAW_DATA_PATH       = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = _PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE         = "phishguard_features.csv"
EDA_DIR             = PROCESSED_DATA_PATH    / "eda"        # EDA artifacts path

MAX_WORKERS   = max(1, cpu_count() - 1)
CSV_CHUNKSIZE = 50_000
PRINT_INTERVAL = 10_000         # log a progress line every N rows
#  Label normalisation map
LABEL_MAP: Dict[str, int] = {
    "spam": 1, "phish": 1, "phishing": 1, "1": 1, "yes": 1, "true": 1, "malicious": 1,
    "ham":  0, "legit": 0, "legitimate": 0, "0": 0, "no":  0, "false": 0,
}
# Phishing-indicative urgency vocabulary
URGENT_WORDS = {
    "urgent", "immediately", "asap", "now", "today", "within 24 hours",
    "limited time", "expires", "deadline", "final notice", "last chance",
    "verify", "verification required", "confirm", "validate",
    "suspended", "suspend", "locked", "blocked", "restricted",
    "unauthorized", "unusual activity", "compromised",
    "security alert", "account alert",
    "password", "login", "sign in", "sign-in", "reset password",
    "update credentials", "re-authenticate",
    "invoice", "payment", "paid", "overdue", "refund",
    "billing", "wire transfer", "gift card",
    "transaction", "purchase", "receipt",
    "legal action", "court", "lawsuit", "law enforcement",
    "irs", "tax", "penalty", "fine",
    "click below", "click here", "open attachment",
    "download attached file", "review document",
}
# Pre-compiled regex patterns (compiled once, reused everywhere)
URL_RE          = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE        = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
DIGIT_RE        = re.compile(r"\d+")
HTML_TAG_RE     = re.compile(r"<.*?>")
MULTI_SPACE_RE  = re.compile(r"\s+")
REPEATED_RE     = re.compile(r"(.)\1{2,}")
IP_URL_RE       = re.compile(r"^https?://(?:\d{1,3}\.){3}\d{1,3}\b")
AUTH_KEYWORDS   = ("authentication-results:", "received-spf", "dkim-signature", "dmarc=")
SPF_RE          = re.compile(r"\bspf=([a-zA-Z0-9_-]+)\b",  re.IGNORECASE)
DKIM_RE         = re.compile(r"\bdkim=([a-zA-Z0-9_-]+)\b", re.IGNORECASE)
DMARC_RE        = re.compile(r"\bdmarc=([a-zA-Z0-9_-]+)\b",re.IGNORECASE)
RECEIVED_RE     = re.compile(r"(?mi)^\s*Received:")
RETURN_PATH_RE  = re.compile(r"(?mi)^Return-Path:\s*<?([^>\r\n]+)>?")
# Auth result encodings
_SPF_FLAGS  = {"pass": 1, "fail": 0, "softfail": 0, "neutral": -1, "none": -1}
_DKIM_FLAGS = {"pass": 1, "fail": 0, "neutral": -1, "none": -1}
_DMARC_FLAGS= {"pass": 1, "fail": 0, "none": -1, "quarantine": 0, "reject": 0}
# Singleton replaced: URL extraction now uses compiled regex (10-50x faster for bulk)
# URLExtract is kept only for production_preprocessing (single-email path) where
# accuracy matters more than throughput.
try:
    from urlextract import URLExtract as _URLExtract
    _URL_EXTRACTOR = _URLExtract()
except ImportError:
    _URL_EXTRACTOR = None

# ........................................................................................
# TEXT UTILITIES
def normalize(text: Optional[str]) -> str:
    """Collapse whitespace and strip; always returns a string."""
    return MULTI_SPACE_RE.sub(" ", str(text or "")).strip()

def is_english(text: str) -> bool:
    """
    Heuristic: drop rows where fewer than 50% of letters are ASCII.
    Avoids re.sub by filtering with str methods directly.
    """
    if not text or len(text) < 3:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True     # numeric/symbol-only content (e.g. invoice) - keep
    return sum(c.isascii() for c in letters) / len(letters) > 0.5

def clean_for_embeddings(text: str) -> str:
    """
    Normalise email body for FastText:
      1. Unescape HTML entities
      2. Strip HTML tags
      3. Replace URLs / emails / digits with placeholder tokens
      4. Remove remaining non-ASCII punctuation
      5. Collapse repeated chars, lowercase, collapse whitespace
    Never returns an empty string - falls back to '<EMPTY>'.
    """
    if not text:
        return "<EMPTY>"
    text = html.unescape(text)
    if re.search(r"<[^>]+>", text):
        try:
            text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        except Exception:
            text = HTML_TAG_RE.sub(" ", text)
    # Replace entities
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = DIGIT_RE.sub(" <NUM> ", text)
    # Strict English Filter: Remove non-alphanumeric (except placeholders)
    text = re.sub(r"[^a-zA-Z0-9\s<>]", " ", text)
    text = REPEATED_RE.sub(r"\1", text)
    text = MULTI_SPACE_RE.sub(" ", text.lower()).strip()
    return text or "<EMPTY>"

def shannon_entropy(s: str) -> float:
    """Shannon entropy of character distribution (bits). Returns 0 for empty input."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def compute_hash(subject: str, body: str, sender: str) -> str:
    """SHA-256 content hash used for deduplication."""
    return hashlib.sha256(
        f"{subject}|{body}|{sender}".encode("utf-8", errors="ignore")
    ).hexdigest()

def normalize_label(value) -> int:
    """Map raw label strings to {0, 1, -1} (-1 = unknown)."""
    if value is None:
        return -1
    return LABEL_MAP.get(str(value).strip().lower(), -1)

# ........................................................................................
# URL & HEADER UTILITIES
def safe_find_urls(text: str, use_extractor: bool = False) -> List[str]:
    """
    Extract URLs from text.
    - use_extractor=False (default, training path): regex only - ~10-50x faster.
    - use_extractor=True  (production path): URLExtract for higher recall on
      obfuscated / bare-domain links, falls back to regex if unavailable.
    """
    if not text:
        return []
    if use_extractor and _URL_EXTRACTOR is not None:
        try:
            clean = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            return _URL_EXTRACTOR.find_urls(clean) or []
        except Exception:
            pass
    return URL_RE.findall(text)

def _match_flag(pattern: re.Pattern, text: str, mapping: dict, default: int = -1) -> int:
    """Extract a single auth result token and map it to an integer."""
    if not text:
        return default
    m = pattern.search(text)
    return mapping.get(m.group(1).lower(), default) if m else default

def parse_auth_from_headers(headers_text: str) -> Dict:
    """
    Parse SPF / DKIM / DMARC results and Return-Path domain from raw header text.
    Returns a dict with safe defaults when headers are absent.
    """
    out = {
        "auth_headers_present": False,
        "spf_result":   -1,
        "dkim_result":  -1,
        "dmarc_result": -1,
        "return_path_domain": "",
        "received_count": 0,
    }
    if not headers_text:
        return out
    lower = headers_text.lower()
    out["auth_headers_present"] = any(k in lower for k in AUTH_KEYWORDS)
    out["spf_result"]   = _match_flag(SPF_RE,  headers_text, _SPF_FLAGS)
    out["dkim_result"]  = _match_flag(DKIM_RE, headers_text, _DKIM_FLAGS)
    out["dmarc_result"] = _match_flag(DMARC_RE,headers_text, _DMARC_FLAGS)
    m = RETURN_PATH_RE.search(headers_text)
    if m:
        addr = parseaddr(m.group(1).strip())[1]
        out["return_path_domain"] = addr.split("@", 1)[1].lower() if "@" in addr else ""
    out["received_count"] = len(RECEIVED_RE.findall(headers_text))
    return out

# ........................................................................................
# CORE FEATURE BUILDER
def build_features(
    subject:      str,
    body:         str,
    sender:       str,
    urls:         List[str],
    html_present: int,
    attachments:  List[str],
    label:        int,
    auth_info:    Optional[Dict] = None,
    header_fields:Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Assemble the full feature dict for one email.
    Returns None if the combined subject+body fails the English check
    (signals the caller to drop the row).
    """
    combined = f"{subject} {body}"
    if not is_english(combined):
        return None

    auth_info     = auth_info     or {}
    header_fields = header_fields or {}
    domains       = [urlparse(u).netloc for u in urls if urlparse(u).netloc]
    cleaned_text  = clean_for_embeddings(combined)
    # Set intersection is faster than iterating for each word
    urgent_count  = sum(1 for w in URGENT_WORDS if w in combined.lower())

    return {
        #  Raw text (for heuristics and re-inspection) 
        "subject":          subject,
        "body":             body,
        "clean_text":       cleaned_text,
        "sender":           sender,
        #  Header fields
        "from_header":      header_fields.get("from_header",      ""),
        "recipient":        header_fields.get("recipient",        ""),
        "return_path":      header_fields.get("return_path",      ""),
        "to_header":        header_fields.get("to_header",        ""),
        "message_id":       header_fields.get("message_id",       ""),
        "x_mailer":         header_fields.get("x_mailer",         ""),
        "x_originating_ip": header_fields.get("x_originating_ip", ""),
        "content_type":     header_fields.get("content_type",     ""),
        # counts for the model
        "urls_count":            len(urls),
        "domains_count":          len(dict.fromkeys(domains)),
        "ip_urls_count":          sum(1 for u in urls if IP_URL_RE.match(u)),
        "attachment_names_count": len(attachments),
        # Raw str kept for heuristics/Audit, not used by the model
        "urls":                  ";".join(urls[:500]),
        "domains":               ";".join(dict.fromkeys(domains)),
        #  Engineered security features 
        "urgent_words_count":    urgent_count,
        "digit_ratio":           sum(c.isdigit() for c in body) / max(len(body), 1),
        "body_entropy":          shannon_entropy(body),
        "html_present":          int(bool(html_present)),
        "attachment_names":      ";".join(attachments) if attachments else "",
        "auth_headers_present":  int(auth_info.get("auth_headers_present", 0)),
        "spf_result":            int(auth_info.get("spf_result",  -1)),
        "dkim_result":           int(auth_info.get("dkim_result", -1)),
        "dmarc_result":          int(auth_info.get("dmarc_result",-1)),
        "return_path_domain":    auth_info.get("return_path_domain", ""),
        "received_count":        int(auth_info.get("received_count", 0)),
        "label":                 int(label)
    }

# ........................................................................................
# MIME PART DECODER (shared by parse_eml and production_preprocessing)
def _decode_payload(part) -> Optional[str]:
    """Decode a MIME part payload to str; tries UTF-8 then latin-1."""
    payload = part.get_payload(decode=True)
    if not payload:
        return None
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeError:
        return payload.decode("latin-1", errors="ignore")

def _extract_header_fields(msg) -> Dict[str, str]:
    """Pull the standard header fields we care about into a clean dict."""
    return {
        "from_header":      normalize(msg.get("From", "")),
        "to_header":        normalize(msg.get("To",   "")),
        "recipient":        normalize(msg.get("Delivered-To", "") or msg.get("Envelope-To", "") or msg.get("To", "")),
        "return_path":      normalize(msg.get("Return-Path",  "")),
        "message_id":       normalize(msg.get("Message-ID",   "")),
        "x_mailer":         normalize(msg.get("X-Mailer",     "") or msg.get("X-Mailing",   "")),
        "x_originating_ip": normalize(msg.get("X-Originating-IP", "")),
        "content_type":     normalize(msg.get("Content-Type", "")),
    }

def _walk_mime(msg) -> tuple:
    """
    Walk all MIME parts and collect:
      body_parts (list[str]), html_present (int), attachments (list[str]),
      html_urls (list[str])  <- hrefs hidden inside anchor tags.
    """
    body_parts = []
    html_present = 0
    attachments = []
    html_urls = []
    for part in msg.walk():
        try:
            ctype = part.get_content_type()
            disp  = str(part.get_content_disposition() or "")
            decoded = _decode_payload(part)
            if decoded is None:
                continue
            if ctype == "text/plain" and "attachment" not in disp:
                body_parts.append(decoded)
            elif ctype == "text/html":
                html_present = 1
                soup = BeautifulSoup(decoded, "html.parser")
                html_urls.extend(a["href"] for a in soup.find_all("a", href=True))
                body_parts.append(soup.get_text(" ", strip=False))
            if part.get_filename():
                attachments.append(part.get_filename())
        except Exception:
            continue

    return body_parts, html_present, attachments, html_urls

# ........................................................................................
# EML FILE PARSING
def parse_eml(path: str) -> Optional[Dict]:
    """Parse a single .eml file into a feature dict. Returns None on failure."""
    try:
        with open(path, "rb") as f:
            msg = email.message_from_bytes(f.read())
    except Exception:
        return None
    subject = normalize(msg.get("Subject", ""))
    sender  = normalize(msg.get("From",    ""))
    header_fields = _extract_header_fields(msg)
    # EML version
    body_parts, html_present, attachments, _ = _walk_mime(msg)
    body = normalize(" ".join(body_parts))
    urls = safe_find_urls(body)
    headers_text = "\n".join(f"{k}: {v}" for k, v in msg.items())
    auth_info = parse_auth_from_headers(headers_text)
    return build_features(subject, body, sender, urls, html_present, attachments, -1, auth_info, header_fields)

# ........................................................................................
# CSV ROW PARSING
def parse_csv_row(row: Dict) -> Optional[Dict]:
    """
    Parse one CSV row into a feature dict.
    Tries multiple column-name variants to handle heterogeneous datasets.
    """
    label = normalize_label(row.get("phish") or row.get("label") or row.get("class") or row.get("spam"))
    header_fields = {
        "from_header":      normalize(row.get("from")            or row.get("from_header")),
        "to_header":        normalize(row.get("to")              or row.get("to_header")),
        "recipient":        normalize(row.get("delivered-to")    or row.get("recipient") or row.get("to")),
        "return_path":      normalize(row.get("return-path")     or row.get("return_path")),
        "message_id":       normalize(row.get("message-id")      or row.get("message_id")),
        "x_mailer":         normalize(row.get("x-mailer")        or row.get("x_mailer")),
        "x_originating_ip": normalize(row.get("x-originating-ip")or row.get("x_originating_ip")),
        "content_type":     normalize(row.get("content-type")    or row.get("content_type")),
    }
    headers_text = str(row.get("raw_headers") or row.get("headers") or "")
    auth_info    = parse_auth_from_headers(headers_text) if headers_text else {}
    # Normalize string fields once
    text_fields = [(k, normalize(v)) for k, v in row.items() if isinstance(v, str)]
    text_fields = [(k, v) for k, v in text_fields if v]   # drop empty after normalize
    if not text_fields:
        return None
    text_fields.sort(key=lambda x: len(x[1]), reverse=True)
    body    = text_fields[0][1]
    subject = text_fields[1][1] if len(text_fields) > 1 else ""
    sender  = normalize(row.get("from") or row.get("sender") or "")
    urls    = safe_find_urls(body)              # fast regex path
    return build_features(subject, body, sender, urls, 0, [], label, auth_info, header_fields)

# ........................................................................................
# DATA LOADING HELPERS
def iter_csv_rows(path: str) -> Iterable[Dict]:
    """
    Yield CSV rows in chunks using the fast C engine.
    The C engine supports on_bad_lines='skip' since pandas 1.3 and is
    3-5x faster than engine='python' for large files.
    """
    try:
        for chunk in pd.read_csv(
            path, dtype=str, engine="c", on_bad_lines="skip",
            chunksize=CSV_CHUNKSIZE, encoding="utf-8", encoding_errors="ignore"):
            yield from chunk.fillna("").to_dict(orient="records")
    except Exception as e:
        logger.error("Failed reading CSV %s: %s", path, e)

def _progress(current: int, total: int, label: str, start: float) -> None:
    """Print an in-place progress line via carriage return."""
    elapsed = time.time() - start
    rate    = current / elapsed if elapsed > 0 else 0
    pct     = 100 * current / total if total else 0
    sys.stderr.write(f"\r  {label}: {current:,}/{total:,}  ({pct:.1f}%)  {rate:,.0f} rows/s   ")
    sys.stderr.flush()

def process_emls(files: List[str]) -> List[Dict]:
    """Parallel EML parsing; logs progress via in-place text lines."""
    if not files:
        return []
    results: List[Dict] = []
    start = time.time()
    total = len(files)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(parse_eml, f): f for f in files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r is not None:
                results.append(r)
            if done % 100 == 0 or done == total:
                _progress(done, total, "EMLs", start)
    sys.stderr.write("\n")
    return results

def _process_chunk(chunk_records: List[Dict]) -> List[Dict]:
    """Worker: parse a list of raw CSV row dicts -> feature dicts. Runs in subprocess."""
    out = []
    for row in chunk_records:
        try:
            rec = parse_csv_row(row)
            if rec is not None:
                out.append(rec)
        except Exception:
            pass
    return out

def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """Walk raw_dir, parse all EML and CSV files, return a combined DataFrame."""
    records: List[Dict] = []
    eml_files, csv_files = [], []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            full = os.path.join(root, f)
            if f.lower().endswith(".eml"):
                eml_files.append(full)
            elif f.lower().endswith(".csv"):
                csv_files.append(full)

    if eml_files:
        logger.info("Parsing %d EML files …", len(eml_files))
        records.extend(process_emls(eml_files))

    for csv_path in csv_files:
        logger.info("Reading CSV: %s", csv_path)
        # Count lines for progress display (fast: read in binary mode)
        try:
            file_rows = sum(1 for _ in open(csv_path, "rb")) - 1
        except Exception:
            file_rows = 0

        # Submit chunks to the process pool for parallel feature extraction
        processed = 0
        start = time.time()
        name  = Path(csv_path).name
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = []
            for chunk in pd.read_csv(
                csv_path, dtype=str, engine="c", on_bad_lines="skip",
                chunksize=CSV_CHUNKSIZE, encoding="utf-8", encoding_errors="ignore"
            ):
                chunk_list = chunk.fillna("").to_dict(orient="records")
                futures.append(ex.submit(_process_chunk, chunk_list))

            for fut in as_completed(futures):
                batch = fut.result()
                records.extend(batch)
                processed += len(batch)
                if file_rows:
                    _progress(processed, file_rows, name, start)
                else:
                    elapsed = time.time() - start
                    rate = processed / elapsed if elapsed > 0 else 0
                    sys.stderr.write(f"\r  {name}: {processed:,} rows  {rate:,.0f} rows/s   ")
                    sys.stderr.flush()
        sys.stderr.write("\n")

    return pd.DataFrame(records)

# ........................................................................................
# EDA (exploratory data analysis) - save plots + summary artifacts
def _save_plot(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("EDA plot saved -> %s", path)

def save_eda_artifacts(df: pd.DataFrame, eda_dir: str) -> None:
    """
    Persist an EDA artifacts for the final preprocessed dataset.
    Saved files:
    eda/summary_stats.csv       - df.describe() for numeric columns
    eda/dtype_info.csv          - column names + dtypes
    eda/null_counts.csv         - per-column null / empty-string counts
    eda/label_distribution.png  - bar chart of class balance
    eda/feature_distributions/  - one histogram per numeric security feature
    eda/top_urgent_words.png    - (if urgent_words_count present) distribution
    eda/body_length_dist.png    - body character-length distribution
    """
    eda_path = Path(eda_dir)
    feat_dir = eda_path / "feature_distributions"
    feat_dir.mkdir(parents=True, exist_ok=True)
    eda_path.mkdir(parents=True, exist_ok=True)
    # 1. Summary stats
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df[numeric_cols].describe().to_csv(str(eda_path / "summary_stats.csv"))
    # 2. Dtypes
    pd.DataFrame({"column": df.columns, "dtype": df.dtypes.values}).to_csv(
        str(eda_path / "dtype_info.csv"), index=False
    )
    # 3. Null + empty-string counts
    null_counts = df.isnull().sum()
    empty_counts = (df == "").sum()
    pd.DataFrame({"null_count": null_counts, "empty_string_count": empty_counts}).to_csv(
        str(eda_path / "null_counts.csv")
    )
    # 4. Label distribution bar chart
    if "label" in df.columns:
        label_counts = df["label"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(
            [str(l) for l in label_counts.index],
            label_counts.values,
            color=["#4caf50", "#f44336", "#9e9e9e"],
        )
        ax.set_title("Label Distribution (0=Legit, 1=Phish, -1=Unknown)")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        for i, v in enumerate(label_counts.values):
            ax.text(i, v + label_counts.max() * 0.01, f"{v:,}", ha="center", fontsize=9)
        _save_plot(fig, str(eda_path / "label_distribution.png"))
    # 5. Numeric feature histograms
    security_features = [
        "urgent_words_count", "digit_ratio", "body_entropy",
        "html_present", "auth_headers_present", "spf_result",
        "dkim_result", "dmarc_result", "received_count",
    ]
    for feat in security_features:
        if feat not in df.columns:
            continue
        series = pd.to_numeric(df[feat], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series, bins=30, edgecolor="black", color="#2196f3", alpha=0.8)
        ax.set_title(f"Distribution: {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        _save_plot(fig, str(feat_dir / f"{feat}.png"))
    # 6. Body length distribution
    if "body" in df.columns:
        lengths = df["body"].astype(str).str.len()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(lengths.clip(upper=lengths.quantile(0.99)), bins=50,
                edgecolor="black", color="#ff9800", alpha=0.85)
        ax.set_title("Body Length Distribution (capped at 99th percentile)")
        ax.set_xlabel("Characters")
        ax.set_ylabel("Count")
        _save_plot(fig, str(eda_path / "body_length_dist.png"))
    logger.info("EDA artifacts saved to %s", eda_dir)

# ........................................................................................
# MAIN TRAINING PIPELINE
def main() -> None:
    start = time.time()
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_PATH / OUTPUT_FILE
    # Incremental: merge with any existing processed output
    old_df = (
        pd.read_csv(output_path, dtype=str).fillna("")
        if output_path.exists() else pd.DataFrame()
    )
    new_df = load_raw_data(RAW_DATA_PATH)
    logger.info("New English records extracted: %d", len(new_df))
    for df_ in (new_df, old_df):
        if "label" not in df_.columns:
            df_["label"] = -1
    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    if combined.empty:
        logger.warning("No data loaded. Exiting.")
        return
    # Deduplication by content hash
    combined["_hash"] = combined.apply(
        lambda r: compute_hash(str(r.get("subject", "")), str(r.get("body", "")), str(r.get("sender", ""))),
        axis=1,
    )
    before = len(combined)
    combined.drop_duplicates("_hash", inplace=True)
    combined.drop(columns=["_hash"], inplace=True)
    logger.info("Deduplication: %d -> %d rows", before, len(combined))
    # Guard: clean_text must never be empty
    if "clean_text" in combined.columns:
        combined["clean_text"] = combined["clean_text"].replace("", "<EMPTY>")
    # Coerce label to int
    try:
        combined["label"] = combined["label"].astype(int)
    except Exception:
        combined["label"] = combined["label"].apply(
            lambda v: int(v) if str(v).lstrip("-").isdigit() else -1
        )
    combined.to_csv(output_path, index=False)
    logger.info(
        "Saved %d rows to %s (%.2fs elapsed)", len(combined), output_path, time.time() - start
    )
    # Save
    save_eda_artifacts(combined, EDA_DIR)

# ........................................................................................
# PRODUCTION SINGLE-EMAIL ENTRY POINT
def production_preprocessing(raw_email: str) -> Optional[Dict]:
    """
    Parse a single raw RFC-2822 email string and return its feature dict.
    Also injects raw metadata needed by the heuristic engine in predictor.py.
    Returns None if parsing fails or the email is not English.
    """
    try:
        msg = email.message_from_string(raw_email)
    except Exception:
        return None

    subject = normalize(msg.get("Subject", ""))
    sender  = normalize(msg.get("From",    ""))
    header_fields = _extract_header_fields(msg)
    body_parts, html_present, attachments, html_urls = _walk_mime(msg)
    body = normalize(" ".join(body_parts))
    # Merge URLExtract hits with explicit anchor hrefs (catches obfuscated links)
    urls = list(set(safe_find_urls(body, use_extractor=True) + html_urls))
    headers_text = "\n".join(f"{k}: {v}" for k, v in msg.items())
    auth_info    = parse_auth_from_headers(headers_text)
    features = build_features(
        subject, body, sender, urls,
        html_present, attachments, label=-1,
        auth_info=auth_info, header_fields=header_fields
    )
    if features is not None:
        # Overwrite with definitive values so predictor.py heuristics see raw data
        features["urls"]        = urls
        features["from_header"] = header_fields["from_header"]
        features["return_path"] = header_fields["return_path"]
        features["raw_text"]    = raw_email
    return features

# ........................................................................................
# CLI
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prod":
        result = production_preprocessing(sys.stdin.read())
        print(result)
    else:
        main()