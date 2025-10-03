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
    if df.empty:
        # Return a skeleton DataFrame with expected columns
        return pd.DataFrame(columns=[
            "Name", "Research summary", "pi history", "student history",
            "key words", "link to lab site", "text", "source_file"
        ])

    # Ensure columns exist
    for col in ["Name", "json_name", "Research summary", "pi history", "student history", "key words", "text", "source_file"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object).where(df[col].notna(), "").astype(str)

    # Display name preference: CSV Name > JSON name
    df["DisplayName"] = df["Name"].where(df["Name"].str.strip() != "", df["json_name"])

    # CSV richness score
    df["__csv_len__"] = (
        df["Research summary"].str.len()
        + df["pi history"].str.len()
        + df["student history"].str.len()
        + df["key words"].str.len()
    )

    # Robust entity key: name_key > short_key > normalized DisplayName
    if "name_key" not in df.columns:
        df["name_key"] = ""
    if "short_key" not in df.columns:
        df["short_key"] = ""

    df["__entity_key__"] = df["name_key"].where(df["name_key"].str.strip() != "", df["short_key"])
    df["__entity_key__"] = df["__entity_key__"].where(
        df["__entity_key__"].str.strip() != "",
        df["DisplayName"].apply(normalize_name)
    )

    def _agg(group: pd.DataFrame) -> pd.Series:
        # Keep the row with most CSV content; merge JSON text/source across dupes
        best = group.sort_values(["__csv_len__", "DisplayName"], ascending=[False, True]).iloc[0].copy()
        # Merge JSON text
        best["text"] = "; ".join(pd.Series([t for t in group["text"] if t.strip()]).dropna().unique())
        # Merge sources
        best["source_file"] = ", ".join(sorted(pd.Series([s for s in group["source_file"] if s.strip()]).dropna().unique()))
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
    # vectorizer
    joblib.dump(vec, VECTORS_DIR / "vectorizer.joblib")
    # matrix
    if not issparse(X):
        X = csr_matrix(X)
    save_npz(VECTORS_DIR / "matrix.npz", X)
    # corpus (without doc column)
    corpus_save = corpus_df.drop(columns=["__doc__"], errors="ignore")
    corpus_save.to_csv(VECTORS_DIR / "corpus.csv", index=False, encoding="utf-8")

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def build_corpus(json_df: pd.DataFrame, profiles_df: pd.DataFrame):
    # Outer join on robust keys; both may be missing in one side, that's ok
    for df in (profiles_df, json_df):
        if "name_key" not in df.columns:
            df["name_key"] = ""
        if "short_key" not in df.columns:
            df["short_key"] = ""

    merged = pd.merge(profiles_df, json_df, how="outer", on=["name_key", "short_key"])

    # ---------- Consolidate duplicates & prefer CSV fields ----------
    merged = consolidate_rows(merged)

    # ---------- Guarantee required columns & safe string dtypes ----------
    required_cols = [
        "Name", "Research summary", "pi history",
        "student history", "key words", "link to lab site",
        "text", "source_file"
    ]
    for col in required_cols:
        if col not in merged.columns:
            merged[col] = ""
        # convert to pure python strings to keep .str happy across pandas versions
        merged[col] = merged[col].astype(object).where(merged[col].notna(), "").astype(str)

    # ---------- Build searchable doc ----------
    def make_doc(row) -> str:
        parts = [
            row["Research summary"], row["pi history"], row["student history"],
            row["key words"], row["text"]
        ]
        return re.sub(r"\s+", " ", " ".join([p for p in parts if p])).strip()

    merged["__doc__"] = merged.apply(make_doc, axis=1).astype(str)

    # ---------- Filter out empties WITHOUT using .str on a bad dtype ----------
    name_stripped = merged["Name"].apply(lambda x: str(x).strip())
    doc_stripped = merged["__doc__"].apply(lambda x: str(x).strip())
    merged = merged[(name_stripped != "") | (doc_stripped != "")].copy()

    # De-dupe again by visible display name (use the first)
    merged = merged.sort_values("Name").drop_duplicates(subset=["Name"], keep="first").reset_index(drop=True)

    # ---------- Vectorize ----------
    vec = TfidfVectorizer(**VEC_OPTS)
    X = vec.fit_transform(merged["__doc__"].values) if len(merged) else csr_matrix((0, 1))

    # ---------- Persist index ----------
    save_index(vec, X, merged)

    return merged, vec, X



# ---------- Ranking ----------
def rank(query: str, vec: TfidfVectorizer, X: csr_matrix) -> np.ndarray:
    if X.shape[0] == 0:
        return np.array([])
    q = vec.transform([query or ""])
    return cosine_similarity(q, X).ravel()


# ---------- UI ----------
st.set_page_config(page_title="PI Matcher", page_icon="🔎", layout="wide")
st.title("🔎 PI Interest Matcher")
st.caption("Paste your research interests. Returns cosine-similarity matches with expandable profiles (CSV first, JSON-backed).")

with st.sidebar:
    st.header("Settings")
    st.markdown("**Data sources (repo-relative):**")
    st.code(f"{JSON_DIR}/\n{CSV_PATH}", language="text")
    st.markdown("**Saved index:**")
    st.code(f"{VECTORS_DIR}/vectorizer.joblib\n{VECTORS_DIR}/matrix.npz\n{VECTORS_DIR}/corpus.csv", language="text")
    force_reload = st.checkbox("Force reload (ignore cache)", value=False)
    topk = st.slider("Top K results", min_value=1, max_value=50, value=DEFAULT_TOPK, step=1)
    st.caption("Vectors are automatically refreshed & saved when the index rebuilds.")

if force_reload:
    st.cache_data.clear()

json_df = load_json_folder(JSON_DIR)
profiles_df = load_profiles_csv(CSV_PATH)
corpus_df, vec, X = build_corpus(json_df, profiles_df)

query = st.text_area(
    "Describe your research interests:",
    placeholder="e.g., operando TEM for energy materials; MXenes; ferroelectrics; ML for microscopy…",
    height=140
)

col_btn, col_dl = st.columns([1, 1])
with col_btn:
    go = st.button("Compare", type="primary")
with col_dl:
    if not corpus_df.empty:
        st.download_button(
            "Download PI index (CSV)",
            data=corpus_df.drop(columns=["__doc__"], errors="ignore").to_csv(index=False).encode("utf-8"),
            file_name="pi_index.csv",
            mime="text/csv"
        )

st.markdown("---")

if go:
    if corpus_df.empty:
        st.warning("No PI profiles found. Ensure `Json/` has JSON files and `profiles_from_json.csv` exists.")
    else:
        sims = rank(query, vec, X)
        if sims.size == 0:
            st.warning("Index is empty after loading; check your CSV/JSON inputs.")
        else:
            order = np.argsort(-sims)[:topk]
            raw_scores = sims[order]

            results = corpus_df.iloc[order].copy()
            results["Cosine score"] = raw_scores

            # Summary table with rank (no normalized score)
            st.subheader("Top matches")
            table_df = results[["Name", "Cosine score"]].reset_index(drop=True)
            table_df.index = np.arange(1, len(table_df) + 1)  # rank index starting at 1
            table_df.index.name = "Rank"
            st.dataframe(table_df, use_container_width=True)

            # Expanders with numbered titles
            st.subheader("Profiles")
            for rank_idx, (_, row) in enumerate(results.iterrows(), start=1):
                name = row["Name"] if isinstance(row["Name"], str) and row["Name"].strip() else "(Unnamed PI)"
                label = f"{rank_idx}) {name} — score: {row['Cosine score']:.4f}"
                with st.expander(label, expanded=False):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        rs = row.get("Research summary", "") or ""
                        pi_hist = row.get("pi history", "") or ""
                        stu_hist = row.get("student history", "") or ""
                        kws = row.get("key words", "") or ""

                        if rs: st.markdown("**Research summary**"); st.write(rs)
                        if pi_hist: st.markdown("**PI history**"); st.write(pi_hist)
                        if stu_hist: st.markdown("**Student history**"); st.write(stu_hist)
                        if kws: st.markdown("**Key words**"); st.write(kws)

                    with c2:
                        lab = str(row.get("link to lab site", "") or "").strip()
                        if lab:
                            st.markdown("**Lab site**")
                            st.markdown(f"[{lab}]({lab})", unsafe_allow_html=True)

                        src = str(row.get("source_file", "") or "")
                        if src:
                            st.caption(f"Source file(s): {src}")

                        with st.popover("Show source JSON text"):
                            st.write((row.get("text") or "").strip() or "(no JSON text)")

st.markdown("---")
st.caption("Duplicates are consolidated via robust name keys (full + first-initial/last). CSV fields are prioritized; vectors persist in ./vectors.")
