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


# ---------- Paths & constants ----------
JSON_DIR = Path("Json")                      # Folder containing PI JSON files
CSV_PATH = Path("profiles_from_json.csv")    # Profiles CSV
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
        rows.append({"name": name.strip(), "text": text, "source_file": p.name})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name", "text", "source_file"])
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
        df = agg.rename(columns={"name": "json_name"})[["name_key", "text", "source_file", "json_name"]]
    return df

def load_profiles_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        # Ensure expected columns even if CSV missing
        out = pd.DataFrame(columns=[
            "Name", "Research summary", "pi history", "student history", "key words", "link to lab site"
        ])
        out["name_key"] = ""
        return out

    df = pd.read_csv(csv_path, encoding="utf-8")
    # Trim whitespace in headers and cells to improve matching
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

    # Ensure required columns exist & are strings
    for col in ["Name", "Research summary", "pi history", "student history", "key words", "link to lab site"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    out["name_key"] = out["Name"].fillna("").map(normalize_name)
    return out[["name_key", "Name", "Research summary", "pi history", "student history", "key words", "link to lab site"]]


# ---------- Similarity ----------
@st.cache_data(show_spinner=False)
def build_corpus(json_df: pd.DataFrame, profiles_df: pd.DataFrame) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    # Outer-join: include CSV-only rows so profiles always show
    merged = pd.merge(
        profiles_df, json_df,
        how="outer",
        on="name_key"
    )

    # Display name preference: CSV Name > JSON name
    merged["Name"] = merged["Name"].where(merged["Name"].notna() & (merged["Name"].str.strip() != ""), merged.get("json_name", ""))
    merged["Name"] = merged["Name"].fillna("").astype(str)

    # Guarantee all display fields exist & are strings
    for col in ["Research summary", "pi history", "student history", "key words", "link to lab site", "text", "source_file"]:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("").astype(str)

    # Combined searchable doc (CSV + cleaned JSON text)
    def make_doc(row) -> str:
        parts = [
            row.get("Research summary", ""),
            row.get("pi history", ""),
            row.get("student history", ""),
            row.get("key words", ""),
            row.get("text", "")
        ]
        return re.sub(r"\s+", " ", " ".join([p for p in parts if isinstance(p, str)])).strip()

    merged["__doc__"] = merged.apply(make_doc, axis=1)

    # Drop rows with no name & no content
    mask_has_name = merged["Name"].str.strip() != ""
    mask_has_doc = merged["__doc__"].str.strip() != ""
    merged = merged[mask_has_name | mask_has_doc].copy()

    # Deduplicate by Name (favor rows that have CSV content)
    merged["__csv_len__"] = (
        merged["Research summary"].str.len()
        + merged["pi history"].str.len()
        + merged["student history"].str.len()
        + merged["key words"].str.len()
    )
    merged = merged.sort_values(["__csv_len__", "Name"], ascending=[False, True]) \
                   .drop_duplicates(subset=["Name"], keep="first") \
                   .reset_index(drop=True)
    merged.drop(columns=["__csv_len__"], inplace=True, errors="ignore")

    vec = TfidfVectorizer(**VEC_OPTS)
    X = vec.fit_transform(merged["__doc__"].values) if len(merged) else np.zeros((0, 1))
    return merged, vec, X

def rank(query: str, vec: TfidfVectorizer, X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.array([])
    q = vec.transform([query or ""])
    sims = cosine_similarity(q, X).ravel()
    return sims


# ---------- UI ----------
st.set_page_config(page_title="PI Matcher", page_icon="🔎", layout="wide")
st.title("🔎 PI Interest Matcher")
st.caption("Paste your research interests. We’ll find the closest-matching PIs and show expandable profiles from the CSV & JSON.")

with st.sidebar:
    st.header("Settings")
    st.markdown("**Data sources (repo-relative):**")
    st.code(f"{JSON_DIR}/\n{CSV_PATH}", language="text")
    force_reload = st.checkbox("Force reload (ignore cache)", value=False)
    topk = st.slider("Top K results", min_value=1, max_value=50, value=DEFAULT_TOPK, step=1)
    st.markdown("---")
    st.markdown("**Scoring**")
    st.caption("We show raw cosine similarity and a min–max normalized score across the shown results.")

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
            # Min–max normalize across the displayed results
            s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
            denom = (s_max - s_min) if (s_max - s_min) > 1e-12 else 1.0
            norm_scores = (raw_scores - s_min) / denom

            results = corpus_df.iloc[order].copy()
            results["Cosine score"] = raw_scores
            results["Normalized score"] = norm_scores

            # Summary table with rank
            st.subheader("Top matches")
            table_df = results[["Name", "Cosine score", "Normalized score"]].reset_index(drop=True)
            table_df.index = np.arange(1, len(table_df) + 1)  # rank index starting at 1
            table_df.index.name = "Rank"
            st.dataframe(table_df, use_container_width=True)

            st.subheader("Profiles")
            for rank_idx, (_, row) in enumerate(results.iterrows(), start=1):
                name = row["Name"] if isinstance(row["Name"], str) and row["Name"].strip() else "(Unnamed PI)"
                label = f"{rank_idx}) {name} — score: {row['Cosine score']:.4f} (norm: {row['Normalized score']:.4f})"
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
st.caption("CSV fields are prioritized for display; JSON is used to enrich matching. Scores are cosine similarity and min–max normalized across the shown results.")
