
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import joblib

# -------------------------
# Config
# -------------------------
JSON_DIR = Path(os.environ.get("JSON_DIR", "Json"))
VECTORS_DIR = Path(os.environ.get("VECTORS_DIR", "vectors"))
CSV_PATH = Path(os.environ.get("CSV_PATH", "profiles_from_json.csv"))
ARTIFACT_PREFIX = VECTORS_DIR / "tfidf"

# Vectorizer options (mirrors prior notebook defaults)
VEC_OPTS = dict(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=2
)

# -------------------------
# Helpers
# -------------------------
URL_PATTERN = re.compile(r"https?://\\S+|www\\.\\S+", re.IGNORECASE)

def _is_noise_key(key: str) -> bool:
    if key is None:
        return False
    key_l = str(key).lower()
    noisy = [
        "url", "href", "link", "img", "image", "thumbnail", "avatar", "icon",
        "script", "style", "button", "view article authors"
    ]
    return any(tok in key_l for tok in noisy)

def _strip_noise(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = URL_PATTERN.sub("", text)
    # Drop sections that contain "sorry"
    if "sorry" in t.lower():
        return ""
    # Collapse whitespace
    return re.sub(r"\\s+", " ", t).strip()

def _collect_strings(obj, parent_key=None) -> List[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_noise_key(k):
                continue
            out.extend(_collect_strings(v, parent_key=k))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_strings(v, parent_key=parent_key))
    else:
        if isinstance(obj, str) and not _is_noise_key(parent_key):
            cleaned = _strip_noise(obj)
            if cleaned:
                out.append(cleaned)
    return out

def json_to_text(json_path: Path) -> str:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        parts = _collect_strings(data)
        return "; ".join(parts)
    except Exception as e:
        return f""

def guess_name_from_filename(p: Path) -> str:
    # Use file stem as a fallback key (e.g., "KUnocic" -> "Unocic", "Xu" -> "Xu")
    stem = p.stem
    # Split camelcase or initials and take last token as probable last name
    tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|$)", stem)
    if tokens:
        return tokens[-1]
    return stem

def normalize_name(s: str) -> str:
    return re.sub(r"\\s+", " ", s.strip().lower())

def build_corpus(json_dir: Path) -> Tuple[List[str], List[str], List[Path]]:
    """Return (names, documents, paths) parallel lists."""
    names, docs, paths = [], [], []
    for p in sorted(json_dir.glob("*.json")):
        text = json_to_text(p)
        if not text:
            continue
        names.append(guess_name_from_filename(p))
        docs.append(text)
        paths.append(p)
    return names, docs, paths

def ensure_vectors(json_dir: Path, vectors_dir: Path, vec_opts: Dict) -> Tuple[TfidfVectorizer, sp.csr_matrix, List[str]]:
    vectors_dir.mkdir(parents=True, exist_ok=True)
    vec_path = ARTIFACT_PREFIX.with_suffix(".pkl")
    mat_path = ARTIFACT_PREFIX.with_suffix(".npz")
    meta_path = vectors_dir / "metadata.json"

    if vec_path.exists() and mat_path.exists() and meta_path.exists():
        vectorizer = joblib.load(vec_path)
        matrix = sp.load_npz(mat_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        names = meta.get("names", [])
        return vectorizer, matrix, names

    names, docs, _ = build_corpus(json_dir)
    if not docs:
        raise RuntimeError(f"No JSON files found in {json_dir.resolve()}")

    vectorizer = TfidfVectorizer(**vec_opts)
    matrix = vectorizer.fit_transform(docs)

    joblib.dump(vectorizer, vec_path)
    sp.save_npz(mat_path, matrix)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"names": names}, f, ensure_ascii=False, indent=2)

    return vectorizer, matrix, names

def load_profiles(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        st.warning(f"CSV file not found at: {csv_path.resolve()}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    # Normalize name key for joining
    if "name" in df.columns:
        df["_name_norm"] = df["name"].map(normalize_name)
    elif "Name" in df.columns:
        df["_name_norm"] = df["Name"].map(normalize_name)
    else:
        # Try to infer a column named like "pi" or similar
        name_col = next((c for c in df.columns if "name" in c.lower()), None)
        if name_col:
            df["_name_norm"] = df[name_col].map(normalize_name)
        else:
            df["_name_norm"] = ""
    return df

def best_row_for_key(df: pd.DataFrame, key: str) -> pd.Series:
    """Find the CSV row that best matches a JSON-derived key (likely last name)."""
    key_norm = normalize_name(key)
    # Direct match
    cand = df[df["_name_norm"].str.contains(key_norm, na=False)]
    if len(cand) == 1:
        return cand.iloc[0]
    if len(cand) > 1:
        # Prefer rows where last token matches last name
        last = key_norm.split()[-1]
        exact_last = cand[cand["_name_norm"].str.endswith(last)]
        if len(exact_last) >= 1:
            return exact_last.iloc[0]
        return cand.iloc[0]
    # No contains match: try last-token only
    last = key_norm.split()[-1]
    if last:
        cand2 = df[df["_name_norm"].str.endswith(last)]
        if len(cand2) >= 1:
            return cand2.iloc[0]
    return pd.Series(dtype=object)

def lookup_profile_sections(row: pd.Series) -> Dict[str, str]:
    # Accept multiple possible header spellings
    def pick(*alts):
        for a in alts:
            if a in row and isinstance(row[a], str) and row[a].strip():
                return row[a].strip()
        return ""

    return {
        "Research Summary": pick("Research summary", "Research Summary", "Research_Summary", "Research summary "),
        "PI History": pick("PI history", "PI History", "Employment History", "Employment history"),
        "Student History": pick("Student history", "Student History"),
        "Key Words": pick("Key Words", "Key words", "Keywords", "KeyWords"),
        "Link to Lab Site": pick("Link to Lab Site", "Lab Link", "Lab Site", "Website", "Link")
    }

def similarity_search(q: str, vectorizer: TfidfVectorizer, matrix: sp.csr_matrix, names: List[str], top_k: int) -> List[Tuple[str, float, int]]:
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, matrix).ravel()
    idx = np.argsort(-sims)[:top_k]
    return [(names[i], float(sims[i]), int(i)) for i in idx if sims[i] > 0]

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="PI Match Finder", page_icon="🔎", layout="wide")

st.title("🔎 PI Match Finder")
st.caption("Build vectors from your **Json/** folder, then search by research interests.")

with st.sidebar:
    st.header("Setup")
    st.write("Paths (override with env vars if needed):")
    st.code(f"JSON_DIR={JSON_DIR.resolve()}\nCSV_PATH={CSV_PATH.resolve()}\nVECTORS_DIR={VECTORS_DIR.resolve()}", language="bash")
    rebuild = st.button("🔄 Rebuild vectors from Json/")

# Load CSV upfront
df_profiles = load_profiles(CSV_PATH)

# Ensure vectors exist (or rebuild if requested)
try:
    if rebuild:
        # Force rebuild by removing old artifacts
        for p in [ARTIFACT_PREFIX.with_suffix(".pkl"), ARTIFACT_PREFIX.with_suffix(".npz"), VECTORS_DIR / "metadata.json"]:
            if Path(p).exists():
                Path(p).unlink()
    vectorizer, matrix, names = ensure_vectors(JSON_DIR, VECTORS_DIR, VEC_OPTS)
    st.success(f"Vector index ready. {matrix.shape[0]} profiles × {matrix.shape[1]} terms.")
except Exception as e:
    st.error(f"Failed to prepare vectors: {e}")
    st.stop()

st.markdown("---")
st.subheader("Compare your interests")
col1, col2 = st.columns([3,1], vertical_alignment="top")
with col1:
    query = st.text_area("Describe your research interests (a few sentences is great):", height=150, placeholder="e.g., in situ/operando electron microscopy for catalysis; ferroelectric thin films; ML for microscopy data...")
with col2:
    k = st.slider("Top K results", min_value=1, max_value=20, value=10, step=1)

go = st.button("Find Matches", type="primary")

if go and query.strip():
    results = similarity_search(query, vectorizer, matrix, names, k)
    if not results:
        st.warning("No non-zero similarity hits. Try a longer or different description.")
    else:
        st.write(f"### Top {len(results)} matches")
        for disp_name, score, idx in results:
            # Try to find the full row in the CSV
            row = best_row_for_key(df_profiles, disp_name) if not df_profiles.empty else pd.Series(dtype=object)
            nice_name = row["name"] if "name" in row else (row.get("Name", disp_name) if isinstance(row, pd.Series) else disp_name)
            sections = lookup_profile_sections(row) if isinstance(row, pd.Series) and not row.empty else {
                "Research Summary": "",
                "PI History": "",
                "Student History": "",
                "Key Words": "",
                "Link to Lab Site": ""
            }

            with st.expander(f"{nice_name} — similarity {score:.3f}"):
                def bullets(title, text):
                    st.markdown(f"**{title}**")
                    if not text:
                        st.write("_(not available in CSV)_")
                        return
                    # Split into bullets if semicolon- or newline-separated
                    parts = re.split(r";|\\n|\\r\\n", text)
                    parts = [p.strip(" -•\\t") for p in parts if p.strip()]
                    if len(parts) <= 1:
                        st.write(f"- {text.strip()}")
                    else:
                        for ptxt in parts:
                            st.write(f"- {ptxt}")

                bullets("Research Summary", sections["Research Summary"])
                bullets("PI History", sections["PI History"])
                bullets("Student History", sections["Student History"])
                bullets("Key Words", sections["Key Words"])

                link = sections["Link to Lab Site"]
                if link:
                    st.markdown(f"**Link to Lab Site:** [{link}]({link})")
                else:
                    st.markdown("**Link to Lab Site:** _(not available in CSV)_")

else:
    st.info("Enter your interests and click **Find Matches** to see results.")

st.markdown("---")
with st.expander("⚙️ Artifacts & How it works"):
    st.markdown(
        """
- Vectors are saved in **vectors/** as:
  - `tfidf.pkl` (the fitted `TfidfVectorizer`)
  - `tfidf.npz` (sparse TF–IDF matrix, one row per JSON file)
  - `metadata.json` (`names` list parallel to matrix rows)
- JSONs are read from **Json/** and cleaned by removing URLs and any section containing the word *sorry*.
- Results are ranked by cosine similarity.
- The details shown in each expandable profile are pulled from **profiles_from_json.csv** by matching names (case-insensitive). Only these sections are displayed: **Research Summary**, **PI History**, **Student History**, **Key Words**, **Link to Lab Site**.
- To force a rebuild, click **Rebuild vectors from Json/** in the sidebar.
"""
    )
