import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# for persisting vectors
from scipy.sparse import csr_matrix, save_npz, issparse
import joblib


# ---------- Paths & constants ----------
JSON_DIR     = Path("Json")                       # PI JSON files live here
CSV_PATH     = Path("profiles_from_json.csv")     # Profiles CSV
VECTORS_DIR  = Path("vectors")                    # Persisted index artifacts
DEFAULT_TOPK = 10

VEC_OPTS = dict(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=1
)

# ---------- Utilities ----------
STRUCTURAL_TOKENS = {
    "http", "https", "www", "mailto", "tel", "pdf", "jpg", "png", "gif",
    "view", "article", "authors", "copyright", "all rights reserved"
}
URL_REGEX = re.compile(r"""https?://\S+|www\.\S+""", re.IGNORECASE)

def normalize_name(name: str) -> str:
    """Robust, case/punct-insensitive key for joining."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKC", name)
    # remove academic titles
    s = re.sub(r"\b(dr|prof|professor)\.?\s+", "", s, flags=re.I)
    # collapse initials like "J. Q." -> "j q"
    s = re.sub(r"\.", " ", s)
    s = s.strip().lower()
    # keep letters, digits, hyphen, apostrophe, and spaces
    s = re.sub(r"[^a-z0-9\s'\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def short_key(name: str) -> str:
    """
    Secondary key: first initial + last name.
    Helps pair 'J. A. LaBean' with 'J. LaBean' or 'James LaBean'.
    """
    n = normalize_name(name)
    if not n:
        return ""
    parts = n.split()
    if not parts:
        return ""
    # last token is usually last name, first token first name/initial
    first = parts[0][:1]
    last  = parts[-1]
    return f"{first} {last}"

def is_trashy_segment(seg: str) -> bool:
    s = seg.strip().lower()
    if not s:
        return True
    if "sorry" in s:
        return True
    if any(tok in s for tok in STRUCTURAL_TOKENS):
        return True
    return False

def flatten_json_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, (list, tuple, set)):
        for x in obj:
            out.extend(flatten_json_strings(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_json_strings(v))
    return out

def clean_text_blocks(blocks: Iterable[str]) -> str:
    cleaned: list[str] = []
    for seg in blocks:
        seg = URL_REGEX.sub(" ", seg)
        seg = re.sub(r"\s+", " ", seg).strip()
        if seg and not is_trashy_segment(seg):
            cleaned.append(seg)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for s in cleaned:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return "; ".join(uniq)


# ---------- Loaders ----------
def load_json_folder(json_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as fh:
                    data = json.loads(fh.read())
            except Exception:
                data = {"raw_text": p.read_text(encoding="utf-8", errors="ignore")}
        # name from JSON or file-stem
        name = data.get("name") if isinstance(data, dict) else None
        if not isinstance(name, str) or not name.strip():
            name = p.stem
        text_blocks = flatten_json_strings(data)
        text = clean_text_blocks(text_blocks)
        rows.append({"json_name": name.strip(), "text": text, "source_file": p.name})

    if not rows:
        return pd.DataFrame(columns=["json_name", "text", "source_file", "name_key", "short_key"])

    df = pd.DataFrame(rows)
    df["name_key"]  = df["json_name"].map(normalize_name)
    df["short_key"] = df["json_name"].map(short_key)
    return df

def load_profiles_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        cols = ["Name", "Research summary", "pi history", "student history", "key words", "link to lab site", "name_key", "short_key"]
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(csv_path, encoding="utf-8")
    # trim headers and cells
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    cmap = {c.lower(): c for c in df.columns}
    def pick(*aliases, default=None):
        for a in aliases:
            if a in cmap:
                return cmap[a]
        return default

    name_col = pick("name", "pi", "full name")
    rs_col   = pick("research summary", "research_summary", "summary")
    pi_col   = pick("pi history", "employment history", "employment_history")
    stu_col  = pick("student history", "student_history")
    kw_col   = pick("key words", "keywords", "key_words")
    link_col = pick("link to lab site", "lab site", "url", "website")

    out = df.copy()
    if name_col and name_col != "Name": out.rename(columns={name_col: "Name"}, inplace=True)
    if rs_col and rs_col != "Research summary": out.rename(columns={rs_col: "Research summary"}, inplace=True)
    if pi_col and pi_col != "pi history": out.rename(columns={pi_col: "pi history"}, inplace=True)
    if stu_col and stu_col != "student history": out.rename(columns={stu_col: "student history"}, inplace=True)
    if kw_col and kw_col != "key words": out.rename(columns={kw_col: "key words"}, inplace=True)
    if link_col and link_col != "link to lab site": out.rename(columns={link_col: "link to lab site"}, inplace=True)

    # ensure required columns & strings
    for col in ["Name", "Research summary", "pi history", "student history", "key words", "link to lab site"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    out["name_key"]  = out["Name"].map(normalize_name)
    out["short_key"] = out["Name"].map(short_key)
    return out[["name_key", "short_key", "Name", "Research summary", "pi history", "student history", "key words", "link to lab site"]]


# ---------- Index build & persist ----------
def consolidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate duplicates after
