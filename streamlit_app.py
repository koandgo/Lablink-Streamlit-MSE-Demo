import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Paths & constants ----------
JSON_DIR = Path("Json")            # Folder containing your example PI json files
CSV_PATH = Path("profiles_from_json.csv")  # Your combined profiles CSV
DEFAULT_TOPK = 10

# Vectorizer options (tuned for short research-interest blurbs)
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
    """Case/punctuation-insensitive key for joining."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKC", name).strip().lower()
    s = re.sub(r"[^a-z0-9\s\.'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_trashy_segment(seg: str) -> bool:
    s = seg.strip().lower()
    if not s:
        return True
    if "sorry" in s:  # discard any whole section that contains 'sorry'
        return True
    if any(tok in s for tok in STRUCTURAL_TOKENS):
        return True
    return False

def flatten_json_strings(obj: Any) -> List[str]:
    """Recursively collect string values from arbitrary JSON."""
    out = []
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
    cleaned: List[str] = []
    for seg in blocks:
        seg = URL_REGEX.sub(" ", seg)              # remove URLs
        seg = re.sub(r"\s+", " ", seg).strip()
        if seg and not is_trashy_segment(seg):
            cleaned.append(seg)
    # de-dup short repeated fragments while preserving order
    seen = set()
    uniq = []
    for s in cleaned:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return "; ".join(uniq)

def load_json_folder(json_dir: Path) -> pd.DataFrame:
    """Build a DF [name, text, source_file] from all JSON files (1 row per PI)."""
    rows = []
    for p in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            # Fallback: try line-delimited JSON or raw text
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
                data = json.loads(txt)
            except Exception:
                data = {"raw_text": p.read_text(encoding="utf-8", errors="ignore")}

        # Heuristic name: prefer top-level 'name' else file stem
        name = data.get("name") if isinstance(data, dict) else None
        if not isinstance(name, str) or not name.strip():
            name = p.stem

        text_blocks = flatten_json_strings(data)
        text = clean_text_blocks(text_blocks)
        rows.append(
            {"name": name.strip(), "text": text, "source_file": p.name}
        )

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name", "text", "source_file"])

    # Ensure one entry per PI: group by normalized name and concatenate unique text
    if not df.empty:
        df["name_key"] = df["name"].map(normalize_name)
        agg = (
            df.groupby(["name_key"], as_index=False)
              .agg({
                  "name": "first",
                  "text": lambda s: "; ".join([x for x in pd.Series(s).dropna().unique() if x]),
                  "source_file": lambda s: ", ".join(sorted(pd.Series(s).dropna().unique()))
              })
        )
        df = agg[["name", "text", "source_file", "name_key"]].copy()
    return df

def load_profiles_csv(csv_path: Path) -> pd.DataFrame:
    """Load the profiles CSV and standardize expected columns, keeping everything else."""
    if not csv_path.exists():
        return pd.DataFrame(columns=[
            "Name", "Research summary", "pi history", "student history",
            "key words", "link to lab site"
        ])

    df = pd.read_csv(csv_path, encoding="utf-8")
    # Build case-insensitive column map
    cmap = {c.lower().strip(): c for c in df.columns}
    # Friendly getters
    def pick(*aliases, default=None):
        for a in aliases:
            if a in cmap:
                return cmap[a]
        return default

    name_col  = pick("name", "pi", "full name", default=None)
    rs_col    = pick("research summary", "research_summary", "summary", default=None)
    pi_col    = pick("pi history", "employment history", "employment_history", default=None)
    stu_col   = pick("student history", "student_history", default=None)
    kw_col    = pick("key words", "keywords", "key_words", default=None)
    link_col  = pick("link to lab site", "lab site", "url", "website", default=None)

    # Preserve all columns but create normalized keys alongside
    out = df.copy()
    if name_col is None:
        out["Name"] = ""
    elif name_col != "Name":
        out.rename(columns={name_col: "Name"}, inplace=True)

    if rs_col and rs_col != "Research summary":
        out.rename(columns={rs_col: "Research summary"}, inplace=True)
    if pi_col and pi_col != "pi history":
        out.rename(columns={pi_col: "pi history"}, inplace=True)
    if stu_col and stu_col != "student history":
        out.rename(columns={stu_col: "student history"}, inplace=True)
    if kw_col and kw_col != "key words":
        out.rename(columns={kw_col: "key words"}, inplace=True)
    if link_col and link_col != "link to lab site":
        out.rename(columns={link_col: "link to lab site"}, inplace=True)

    # Ensure required columns exist
    for col in ["Research summary", "pi history", "student history", "key words", "link to lab site"]:
        if col not in out.columns:
            out[col] = ""

    out["name_key"] = out["Name"].fillna("").map(normalize_name)
    return out


# ---------- Similarity ----------
@st.cache_data(show_spinner=False)
def build_corpus(json_df: pd.DataFrame, profiles_df: pd.DataFrame) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """Join JSON text with CSV fields for a richer search corpus; return matrix and vectorizer."""
    # Merge on normalized name
    merged = pd.merge(
        json_df, profiles_df,
        how="outer",
        on="name_key",
        suffixes=("_json", "_csv")
    )

    # Final display Name preference: CSV Name > JSON name
    merged["Name"] = merged["Name"].fillna(merged["name"])
    merged["Name"] = merged["Name"].fillna("").astype(str)
    merged["Name"] = merged["Name"].where(merged["Name"].str.strip() != "", merged["name"])

    # Build a combined searchable text field
    def make_doc(row) -> str:
        parts = [
            str(row.get("text", "") or ""),
            str(row.get("Research summary", "") or ""),
            str(row.get("pi history", "") or ""),
            str(row.get("student history", "") or ""),
            str(row.get("key words", "") or "")
        ]
        # Avoid overly long whitespace
        doc = " ".join(p.strip() for p in parts if isinstance(p, str))
        doc = re.sub(r"\s+", " ", doc)
        return doc.strip()

    merged["__doc__"] = merged.apply(make_doc, axis=1)
    # Deduplicate rows by Name (keep first)
    merged = merged.sort_values("Name").drop_duplicates(subset=["Name"], keep="first").reset_index(drop=True)

    # Vectorize
    vec = TfidfVectorizer(**VEC_OPTS)
    X = vec.fit_transform(merged["__doc__"].values)

    return merged, vec, X

def rank(query: str, vec: TfidfVectorizer, X: np.ndarray) -> np.ndarray:
    q = vec.transform([query or ""])
    sims = cosine_similarity(q, X).ravel()
    return sims


# ---------- UI ----------
st.set_page_config(page_title="PI Matcher", page_icon="🔎", layout="wide")

st.title("🔎 PI Interest Matcher")
st.caption("Paste your research interests. We’ll find the closest-matching PIs and show expandable profiles.")

# Sidebar: configuration
with st.sidebar:
    st.header("Settings")
    st.markdown("**Data sources (repo-relative):**")
    st.code(f"{JSON_DIR}/\n{CSV_PATH}", language="text")

    # Reload toggle
    force_reload = st.checkbox("Force reload (ignore cache)", value=False)

    topk = st.slider("Top K results", min_value=1, max_value=50, value=DEFAULT_TOPK, step=1)

    st.markdown("---")
    st.markdown("**Vectorizer**")
    st.caption("TF-IDF (1–2 grams, english stopwords)")

# Load data
if force_reload:
    st.cache_data.clear()

json_df = load_json_folder(JSON_DIR)
profiles_df = load_profiles_csv(CSV_PATH)
corpus_df, vec, X = build_corpus(json_df, profiles_df)

# Main input
query = st.text_area(
    "Describe your research interests:",
    placeholder="e.g., cryo/in situ electron microscopy for operando catalysis; MXenes; ferroelectric oxide membranes; machine learning for microscopy…",
    height=140
)

col_btn, col_dl = st.columns([1, 1])
with col_btn:
    go = st.button("Compare", type="primary")
with col_dl:
    if not corpus_df.empty:
        st.download_button(
            "Download PI index (CSV)",
            data=corpus_df.drop(columns=["__doc__"]).to_csv(index=False).encode("utf-8"),
            file_name="pi_index.csv",
            mime="text/csv"
        )

st.markdown("---")

if go:
    if corpus_df.empty:
        st.warning("No PI profiles were found. Ensure `Json/` has JSON files and `profiles_from_json.csv` exists.")
    else:
        sims = rank(query, vec, X)
        order = np.argsort(-sims)[:topk]
        results = corpus_df.iloc[order].copy()
        results["Similarity score"] = (sims[order]).round(4)

        # Summary table
        st.subheader("Top matches")
        st.dataframe(
            results[["Name", "Similarity score"]].reset_index(drop=True),
            use_container_width=True
        )

        # Expanders
        st.subheader("Profiles")
        for _, row in results.iterrows():
            with st.expander(f"{row['Name']}  —  score: {row['Similarity score']:.4f}", expanded=False):
                # Two-column layout inside expander
                c1, c2 = st.columns([2, 1])

                with c1:
                    rs = row.get("Research summary", "") or ""
                    pi_hist = row.get("pi history", "") or ""
                    stu_hist = row.get("student history", "") or ""
                    kws = row.get("key words", "") or ""

                    if rs:
                        st.markdown("**Research summary**")
                        st.write(rs)

                    if pi_hist:
                        st.markdown("**PI history**")
                        st.write(pi_hist)

                    if stu_hist:
                        st.markdown("**Student history**")
                        st.write(stu_hist)

                    if kws:
                        st.markdown("**Key words**")
                        st.write(kws)

                with c2:
                    lab = str(row.get("link to lab site", "") or "").strip()
                    if lab:
                        st.markdown("**Lab site**")
                        st.markdown(f"[{lab}]({lab})", unsafe_allow_html=True)

                    src = str(row.get("source_file", "") or "")
                    if src:
                        st.caption(f"Source file(s): {src}")

                    with st.popover("Show source JSON text"):
                        # show the cleaned text used in the search index
                        st.write((row.get("text") or "").strip() or "(no text)")

        st.info("Tip: adjust **Top K results** in the sidebar or edit your interest text, then click **Compare** again.")


# Footer
st.markdown("---")
st.caption("Built for comparing student research interests against PI profiles from JSON + CSV.")
