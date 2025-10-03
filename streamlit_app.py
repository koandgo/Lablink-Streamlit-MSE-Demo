import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, issparse, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================
# Paths & constants
# =============================
JSON_DIR     = Path("Json")                       # Folder with PI JSON files
CSV_PATH     = Path("profiles_from_json.csv")     # Profiles CSV
VECTORS_DIR  = Path("vectors")                    # Where index artifacts are saved
DEFAULT_TOPK = 10

# TF-IDF config (sane defaults for short research blurbs)
VEC_OPTS = dict(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=1
)

# =============================
# Text cleaning & keying
# =============================
STRUCTURAL_TOKENS = {
    "http", "https", "www", "mailto", "tel", "pdf", "jpg", "png", "gif",
    "view", "article", "authors", "copyright", "all rights reserved"
}
URL_REGEX = re.compile(r"""https?://\S+|www\.\S+""", re.IGNORECASE)

def normalize_name(name: str) -> str:
    """Robust, case/punct-insensitive key for joining CSV↔JSON."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKC", name)
    s = re.sub(r"\b(dr|prof|professor)\.?\s+", "", s, flags=re.I)  # drop titles
    s = s.replace(".", " ")
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s'\-]", " ", s)  # keep letters/digits/space/'/-
    s = re.sub(r"\s+", " ", s).strip()
    return s

def short_key(name: str) -> str:
    """Secondary key: first-initial + last-name (e.g., 'j laban' for 'J. A. LaBean')."""
    n = normalize_name(name)
    if not n:
        return ""
    parts = n.split()
    if not parts:
        return ""
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


# =============================
# Loaders
# =============================
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
    # Trim headers and cells to avoid silent mismatches
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    cmap = {c.lower(): c for c in df.columns}
    def pick(*aliases, default=None):
        for a in aliases:
            if a in cmap: return cmap[a]
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

    # Ensure columns exist and are strings
    for col in ["Name", "Research summary", "pi history", "student history", "key words", "link to lab site"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    out["name_key"]  = out["Name"].map(normalize_name)
    out["short_key"] = out["Name"].map(short_key)
    return out[["name_key", "short_key", "Name", "Research summary", "pi history", "student history", "key words", "link to lab site"]]


# =============================
# Consolidation & Index
# =============================
def consolidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate duplicates after merging CSV & JSON.
    Prefer CSV fields for display; merge JSON text & sources.
    Output: one row per consolidated person.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "Name", "Research summary", "pi history", "student history",
            "key words", "link to lab site", "text", "source_file"
        ])

    # Ensure columns exist and are strings
    for col in ["Name", "json_name", "Research summary", "pi history", "student history", "key words", "text", "source_file"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object).where(df[col].notna(), "").astype(str)

    # Display name preference: CSV Name > JSON name
    df["DisplayName"] = df["Name"].where(df["Name"].str.strip() != "", df["json_name"])

    # CSV richness to pick the best row
    df["__csv_len__"] = (
        df["Research summary"].str.len()
        + df["pi history"].str.len()
        + df["student history"].str.len()
        + df["key words"].str.len()
    )

    # Robust entity key: name_key > short_key > normalized DisplayName
    for col in ["name_key", "short_key"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    df["__entity_key__"] = df["name_key"].where(df["name_key"].str.strip() != "", df["short_key"])
    df["__entity_key__"] = df["__entity_key__"].where(
        df["__entity_key__"].str.strip() != "",
        df["DisplayName"].apply(normalize_name)
    )

    def _agg(group: pd.DataFrame) -> pd.Series:
        best = group.sort_values(["__csv_len__", "DisplayName"], ascending=[False, True]).iloc[0].copy()
        # Merge JSON text across dupes
        all_texts = [t for t in group["text"] if isinstance(t, str) and t.strip()]
        best["text"] = "; ".join(pd.Series(all_texts).dropna().unique())
        # Merge sources
        all_srcs = [s for s in group["source_file"] if isinstance(s, str) and s.strip()]
        best["source_file"] = ", ".join(sorted(pd.Series(all_srcs).dropna().unique()))
        return best

    out = (
        df.groupby("__entity_key__", as_index=False, sort=False)
          .apply(_agg)
          .reset_index(drop=True)
    )

    out.rename(columns={"DisplayName": "Name"}, inplace=True)
    out = out[[
        "Name", "Research summary", "pi history", "student history",
        "key words", "link to lab site", "text", "source_file"
    ]]
    return out

def ensure_vectors_dir():
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

def save_index(vec: TfidfVectorizer, X: csr_matrix, corpus_df: pd.DataFrame):
    ensure_vectors_dir()
    # Vectorizer
    joblib.dump(vec, VECTORS_DIR / "vectorizer.joblib")
    # Matrix
    if not issparse(X):
        X = csr_matrix(X)
    save_npz(VECTORS_DIR / "matrix.npz", X)
    # Corpus (no __doc__)
    corpus_df.drop(columns=["__doc__"], errors="ignore").to_csv(
        VECTORS_DIR / "corpus.csv", index=False, encoding="utf-8"
    )

def build_docs(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Create the searchable '__doc__' field."""
    for col in ["Research summary", "pi history", "student history", "key words", "text"]:
        if col not in corpus_df.columns:
            corpus_df[col] = ""
        corpus_df[col] = corpus_df[col].astype(object).where(corpus_df[col].notna(), "").astype(str)

    def make_doc(row) -> str:
        parts = [row["Research summary"], row["pi history"], row["student history"], row["key words"], row["text"]]
        return re.sub(r"\s+", " ", " ".join([p for p in parts if p])).strip()

    corpus_df["__doc__"] = corpus_df.apply(make_doc, axis=1).astype(str)
    return corpus_df

def rebuild_index(json_df: pd.DataFrame, profiles_df: pd.DataFrame) -> Tuple[pd.DataFrame, TfidfVectorizer, csr_matrix]:
    # Ensure join keys exist
    for df in (profiles_df, json_df):
        if "name_key" not in df.columns: df["name_key"] = ""
        if "short_key" not in df.columns: df["short_key"] = ""

    merged = pd.merge(profiles_df, json_df, how="outer", on=["name_key", "short_key"])

    corpus_df = consolidate_rows(merged)
    corpus_df = build_docs(corpus_df)

    # Filter out empty rows WITHOUT relying on .str accessor
    name_stripped = corpus_df["Name"].apply(lambda x: str(x).strip())
    doc_stripped  = corpus_df["__doc__"].apply(lambda x: str(x).strip())
    corpus_df = corpus_df[(name_stripped != "") | (doc_stripped != "")].copy()

    # De-dupe by display name
    corpus_df = corpus_df.sort_values("Name").drop_duplicates(subset=["Name"], keep="first").reset_index(drop=True)

    # Vectorize
    vec = TfidfVectorizer(**VEC_OPTS)
    X = vec.fit_transform(corpus_df["__doc__"].values) if len(corpus_df) else csr_matrix((0, 1))

    # Persist brand-new index
    save_index(vec, X, corpus_df)
    return corpus_df, vec, X

def rank(query: str, vec: TfidfVectorizer, X: csr_matrix) -> np.ndarray:
    if X.shape[0] == 0:
        return np.array([])
    q = vec.transform([query or ""])
    return cosine_similarity(q, X).ravel()


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="PI Matcher (Rebuilt)", page_icon="🔎", layout="wide")
st.title("🔎 PI Interest Matcher — Fresh Rebuild")
st.caption("This app rebuilds the search index from `Json/` and `profiles_from_json.csv`, then saves it to `vectors/`.")

with st.sidebar:
    st.header("Settings")
    st.markdown("**Inputs (repo-relative):**")
    st.code(f"{JSON_DIR}/\n{CSV_PATH}", language="text")
    st.markdown("**Index artifacts (saved):**")
    st.code("vectors/vectorizer.joblib\nvectors/matrix.npz\nvectors/corpus.csv", language="text")

    # Always rebuild per your request; button is still useful to force a re-run.
    force_rebuild = st.checkbox("Force rebuild now", value=True)
    topk = st.slider("Top K results", 1, 50, DEFAULT_TOPK, 1)

# Load sources
json_df = load_json_folder(JSON_DIR)
profiles_df = load_profiles_csv(CSV_PATH)

# Always rebuild (per your instruction), but keep a cache so reruns in-session are fast
@st.cache_data(show_spinner=True)
def _rebuild_cached(_json_df: pd.DataFrame, _profiles_df: pd.DataFrame):
    return rebuild_index(_json_df, _profiles_df)

if force_rebuild:
    st.cache_data.clear()

corpus_df, vec, X = _rebuild_cached(json_df, profiles_df)

# Input
query = st.text_area(
    "Describe your research interests:",
    placeholder="e.g., operando TEM for energy materials; MXenes; ferroelectrics; ML for microscopy…",
    height=140
)

# Actions
col_btn, col_dl, col_dbg = st.columns([1, 1, 1])
with col_btn:
    go = st.button("Compare", type="primary")
with col_dl:
    if not corpus_df.empty:
        st.download_button(
            "Download deduped corpus (CSV)",
            data=corpus_df.drop(columns=["__doc__"], errors="ignore").to_csv(index=False).encode("utf-8"),
            file_name="corpus_dedup.csv",
            mime="text/csv"
        )
with col_dbg:
    with st.popover("Quick stats"):
        st.write("JSON rows:", len(json_df))
        st.write("CSV rows:", len(profiles_df))
        st.write("Corpus rows:", len(corpus_df))

st.markdown("---")

# Results
if go:
    if corpus_df.empty:
        st.warning("No PI profiles found. Confirm `Json/` has JSON files and `profiles_from_json.csv` exists.")
    else:
        sims = rank(query, vec, X)
        order = np.argsort(-sims)[:topk]
        raw_scores = sims[order]
        results = corpus_df.iloc[order].copy()
        results["Cosine score"] = raw_scores

        # Table
        st.subheader("Top matches")
        table_df = results[["Name", "Cosine score"]].reset_index(drop=True)
        table_df.index = np.arange(1, len(table_df) + 1)
        table_df.index.name = "Rank"
        st.dataframe(table_df, use_container_width=True)

        # Expanders (numbered; no normalized score)
        st.subheader("Profiles")
        for rank_idx, (_, row) in enumerate(results.iterrows(), start=1):
            name = row["Name"] if isinstance(row["Name"], str) and row["Name"].strip() else "(Unnamed PI)"
            label = f"{rank_idx}) {name} — score: {row['Cosine score']:.4f}"
            with st.expander(label, expanded=False):
                c1, c2 = st.columns([2, 1])
                with c1:
                    if row.get("Research summary", ""):
                        st.markdown("**Research summary**")
                        st.write(row["Research summary"])
                    if row.get("pi history", ""):
                        st.markdown("**PI history**")
                        st.write(row["pi history"])
                    if row.get("student history", ""):
                        st.markdown("**Student history**")
                        st.write(row["student history"])
                    if row.get("key words", ""):
                        st.markdown("**Key words**")
                        st.write(row["key words"])

                with c2:
                    lab = str(row.get("link to lab site", "") or "").strip()
                    if lab:
                        st.markdown("**Lab site**")
                        st.markdown(f"[{lab}]({lab})", unsafe_allow_html=True)
                    src = str(row.get("source_file", "") or "").strip()
                    if src:
                        st.caption(f"Source JSON file(s): {src}")

                    with st.popover("Show cleaned JSON text"):
                        st.write((row.get("text") or "").strip() or "(no JSON text)")

st.markdown("---")
st.caption("Rebuilt each run. CSV fields prioritized; JSON enriches search. Index artifacts saved to ./vectors.")
