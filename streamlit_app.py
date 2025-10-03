# app.py
import json, re, os, io
from pathlib import Path
from typing import Any, Iterable, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ---------- Config ----------
JSON_DIR_DEFAULT = Path("Json")            # <- folder of PI JSONs
VECTORS_DIR_DEFAULT = Path("vectors")      # <- artifacts go here
CSV_PROFILE_PATH_DEFAULT = Path("profiles_from_json.csv")

VEC_OPTS = dict(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2,
)

# ---------- Utilities ----------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
DASH_LINE_RE = re.compile(r"^\s*-{2,}\s*$", re.M)

def clean_text(s: str) -> str:
    s = URL_RE.sub(" ", s)
    s = DASH_LINE_RE.sub(" ", s)
    s = re.sub(r"\bview article authors\b", " ", s, flags=re.I)
    s = re.sub(r"\bsorry\b.*?(?:\n\n|\Z)", " ", s, flags=re.I | re.S)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _iter_strings(o: Any) -> Iterable[str]:
    """Yield strings from arbitrary JSON recursively, skipping structural keys commonly noisy."""
    if o is None:
        return
    if isinstance(o, str):
        yield o
    elif isinstance(o, (int, float)):
        yield str(o)
    elif isinstance(o, dict):
        for k, v in o.items():
            if str(k).lower() in {"url", "href", "src", "image", "images", "links", "authors_link"}:
                continue
            yield from _iter_strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _iter_strings(v)

def json_to_text(p: Path) -> Tuple[str, str]:
    """Return (name, text) from a JSON file path."""
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Heuristic name: prefer explicit "name"/"pi"/"title" fields; fallback to filename
    candidates = []
    if isinstance(data, dict):
        for key in ("name", "pi", "title", "pi_name", "full_name"):
            v = data.get(key)
            if isinstance(v, str) and 2 <= len(v) < 200:
                candidates.append(v)
    name = (candidates[0] if candidates else p.stem).strip()

    parts = list(_iter_strings(data))
    text = clean_text("; ".join([t for t in parts if t and not URL_RE.search(t)]))
    return name, text

def build_corpus(json_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(json_dir.glob("*.json")):
        try:
            name, text = json_to_text(p)
            if text:
                rows.append({"name": name, "text": text, "source_file": p.name})
        except Exception as e:
            st.warning(f"Failed to parse {p.name}: {e}")
    df = pd.DataFrame(rows).drop_duplicates(subset=["name"], keep="first")
    return df

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def save_artifacts(vec_dir: Path, X: csr_matrix, vectorizer: TfidfVectorizer, names: List[str], texts_sha: str):
    ensure_dir(vec_dir)
    save_npz(vec_dir / "tfidf_vectors.npz", X)
    joblib.dump(vectorizer, vec_dir / "tfidf_vectorizer.pkl")
    pd.Series(names).to_csv(vec_dir / "names.csv", index=False, header=False)
    with open(vec_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"texts_digest": texts_sha, "count": len(names)}, f)

def load_artifacts(vec_dir: Path) -> Tuple[csr_matrix, TfidfVectorizer, List[str]]:
    X = load_npz(vec_dir / "tfidf_vectors.npz")
    vectorizer: TfidfVectorizer = joblib.load(vec_dir / "tfidf_vectorizer.pkl")
    names = [line.strip() for line in (vec_dir / "names.csv").read_text(encoding="utf-8").splitlines()]
    return X, vectorizer, names

def quick_digest(texts: Iterable[str]) -> str:
    # Lightweight, deterministic digest (no hashlib to keep dependencies slim)
    acc = 1469598103934665603  # FNV-1a 64-bit offset basis
    for t in texts:
        for b in t.encode("utf-8"):
            acc ^= b
            acc = (acc * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{acc:016x}"

def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

def load_profiles(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    dfp = pd.read_csv(csv_path)
    # Standardize expected column names if present
    rename_map = {
        "Name": "name",
        "Research summary": "research_summary",
        "Research Summary": "research_summary",
        "Employment History": "pi_history",
        "PI History": "pi_history",
        "Student History": "student_history",
        "Key Words": "key_words",
        "Keywords": "key_words",
        "Link to Lab Site": "link_to_lab",
        "Lab Site": "link_to_lab",
    }
    for k, v in rename_map.items():
        if k in dfp.columns and v not in dfp.columns:
            dfp[v] = dfp[k]
    need = ["name", "research_summary", "pi_history", "student_history", "key_words", "link_to_lab"]
    for col in need:
        if col not in dfp.columns:
            dfp[col] = ""
    dfp["name_norm"] = dfp["name"].map(normalize_name)
    return dfp[["name", "name_norm", "research_summary", "pi_history", "student_history", "key_words", "link_to_lab"]]

# ---------- UI ----------
st.set_page_config(page_title="PI Match Finder", page_icon="🔎", layout="wide")
st.title("🔎 PI Match Finder")
st.caption("Paste your research interests to find the best-matching PIs, then click a name to expand the profile.")

# Paths (allow overrides in sidebar)
with st.sidebar:
    st.header("Settings")
    json_dir = Path(st.text_input("JSON folder", value=str(JSON_DIR_DEFAULT)))
    vectors_dir = Path(st.text_input("Vectors folder", value=str(VECTORS_DIR_DEFAULT)))
    csv_profile_path = Path(st.text_input("Profiles CSV path", value=str(CSV_PROFILE_PATH_DEFAULT)))
    top_k = st.slider("Top-K results", min_value=1, max_value=20, value=10, step=1)
    reindex = st.button("Rebuild vectors from JSON")

# Build / Load index
status_placeholder = st.empty()

@st.cache_resource(show_spinner=True)
def _build_index(json_dir: Path, vectors_dir: Path) -> Tuple[pd.DataFrame, csr_matrix, TfidfVectorizer, List[str]]:
    df = build_corpus(json_dir)
    if df.empty:
        raise RuntimeError(f"No JSON files found in {json_dir.resolve()}")
    texts = df["text"].tolist()
    names = df["name"].tolist()
    vec = TfidfVectorizer(**VEC_OPTS)
    X = vec.fit_transform(texts)
    save_artifacts(vectors_dir, X, vec, names, quick_digest(texts))
    return df, X, vec, names

def build_or_load(json_dir: Path, vectors_dir: Path) -> Tuple[pd.DataFrame, csr_matrix, TfidfVectorizer, List[str]]:
    if reindex or not (vectors_dir / "tfidf_vectors.npz").exists():
        status_placeholder.info("Building vectors from JSON…")
        df, X, vec, names = _build_index(json_dir, vectors_dir)
        status_placeholder.success(f"Indexed {len(names)} profiles.")
        return df, X, vec, names
    else:
        status_placeholder.info("Loading existing vectors…")
        try:
            X, vec, names = load_artifacts(vectors_dir)
        except Exception:
            # Fallback: rebuild if artifacts are inconsistent
            df, X, vec, names = _build_index(json_dir, vectors_dir)
            status_placeholder.success(f"Rebuilt {len(names)} profiles.")
            return df, X, vec, names
        # If loaded, we still need the texts & mapping; rebuild a lightweight DF
        df = build_corpus(json_dir)  # to align names with files; if mismatch, we trust vectors' name order
        if not df.empty:
            # align df to names order where possible
            name_to_row = {normalize_name(n): r for r, n in df["name"].map(normalize_name).items()}
            order_rows = []
            for n in names:
                r = name_to_row.get(normalize_name(n))
                if r is not None:
                    order_rows.append(df.iloc[r])
            if order_rows:
                df = pd.DataFrame(order_rows)
        status_placeholder.success(f"Loaded {len(names)} profiles.")
        return df, X, vec, names

df_json, X, vectorizer, names = build_or_load(json_dir, vectors_dir)

# Load profile CSV
profiles_df = load_profiles(csv_profile_path)
profiles_map: Dict[str, Dict[str, str]] = (
    profiles_df.set_index("name_norm").to_dict(orient="index") if not profiles_df.empty else {}
)

# Query box
st.subheader("Your interests")
user_text = st.text_area(
    "Paste a brief description of your research interests (a paragraph or two):",
    height=160,
    placeholder="e.g., in-situ/operando electron microscopy for energy materials; STEM-EELS/EDS; ML for microscopy…",
)
col_run, col_k = st.columns([1, 3])
with col_run:
    run = st.button("Find Matches", type="primary")

# Results# Results
if run and user_text.strip():
    q_clean = clean_text(user_text.strip())
    q_vec = vectorizer.transform([q_clean])
    sims = cosine_similarity(q_vec, X).ravel()
    idx = np.argsort(-sims)[:top_k]

    st.subheader(f"Top {top_k} matches")
    for rank, i in enumerate(idx, start=1):
        name_i = names[i]
        score = float(sims[i])

        # strict CSV-only profile fields
        prof = profiles_map.get(normalize_name(name_i), {})
        research_summary = prof.get("research_summary", "")
        pi_history       = prof.get("pi_history", "")
        student_history  = prof.get("student_history", "")
        key_words        = prof.get("key_words", "")
        link_to_lab      = prof.get("link_to_lab", "")

        # 1) Numbered header like "1) Tracy — similarity: 0.026"
        header = f"{rank}) {name_i} — similarity: {score:.3f}"
        with st.expander(header, expanded=(rank == 1)):
            if research_summary:
                st.markdown("**Research Summary**")
                st.write(research_summary)

            if pi_history:
                st.markdown("**PI History**")
                st.write(pi_history)

            if student_history:
                st.markdown("**Student History**")
                st.write(student_history)

            if key_words:
                st.markdown("**Key Words**")
                st.write(key_words)

            if link_to_lab:
                st.markdown(f"**Link to Lab Site**")
                st.markdown(f"[{link_to_lab}]({link_to_lab})")

    st.caption("Tip: click any name to toggle its profile.")
else:
    st.info("Enter your interests above and click **Find Matches**.")

# Footer
st.markdown("---")
st.caption(
    "JSON → TF-IDF vectors with scikit-learn; profiles enriched from `profiles_from_json.csv`. "
    "Artifacts saved in `vectors/` as `tfidf_vectors.npz` + `tfidf_vectorizer.pkl` + `names.csv`."
)
