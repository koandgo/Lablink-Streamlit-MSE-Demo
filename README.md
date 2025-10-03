# LabLink Streamlit MSE Demo

A Streamlit app that:
- Builds TF–IDF vectors from PI JSON files in `Json/`.
- Lets users paste a research-interest paragraph and returns the top‑K matching PIs.
- Shows expandable PI profiles with **Research Summary**, **PI History**, **Student History**, **Key Words**, and **Link to Lab Site**, pulled from `profiles_from_json.csv`.

## 🚀 Quickstart

### 1) Clone & prepare
```bash
git clone https://github.com/koandgo/lablink-streamlit-mse-demo.git
cd lablink-streamlit-mse-demo
python -m venv .venv && source .venv/bin/activate  # (Win: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Add your data
- Put PI JSON files into `Json/` .
- Place `profiles_from_json.csv` at repo root. The app will read only the following fields:
  - `name` (or `Name`)
  - `Research Summary` (or common variants)
  - `PI History` (or `Employment History`)
  - `Student History`
  - `Key Words`
  - `Link to Lab Site`

> The app is forgiving with header spellings (e.g., `Research summary`, `Employment history`, `Keywords`, etc.).

### 3) Run
```bash
streamlit run streamlit_app.py
```
- First run builds vectors under `vectors/` (`tfidf.pkl`, `tfidf.npz`, `metadata.json`).
- Use the sidebar **Rebuild vectors from Json/** when you add/modify JSONs.

### 4) Deploy on Streamlit Community Cloud
- Push this repo to GitHub.
- On Streamlit Cloud, point to `streamlit_app.py`.
- Optional environment variables (see below).

---

## 📁 Repository structure
```
lablink-streamlit-mse-demo/
├─ streamlit_app.py           # main app (provided)
├─ profiles_from_json.csv     # your CSV (not tracked here)
├─ Json/                      # PI JSON inputs (one per PI)
│  └─ .gitkeep
├─ vectors/                   # TF–IDF artifacts (created at runtime)
│  └─ .gitkeep
├─ requirements.txt
├─ .gitignore
└─ .streamlit/
   └─ config.toml
```

## 🔧 Environment variables
You can override default paths without touching code:
- `JSON_DIR` (default: `Json`)
- `CSV_PATH` (default: `profiles_from_json.csv`)
- `VECTORS_DIR` (default: `vectors`)

On Streamlit Cloud, set these in **App settings → Secrets** as environment variables.

## 🧠 How it works
- Reads all `*.json` files in `JSON_DIR`, removes URLs and sections containing “sorry”, and concatenates found strings.
- Builds a TF–IDF matrix (unigram+bigram, English stopwords, `min_df=2`, `max_df=0.9`).
- Ranks cosine similarity between the user query and each PI document.
- Looks up the best CSV row by name (case-insensitive); displays only the whitelisted sections in bullets.
- Persists the fitted vectorizer & matrix in `VECTORS_DIR`.

## 🧪 CSV schema tips
The app looks for common header variants. A safe header set is:
```
name,Research Summary,PI History,Student History,Key Words,Link to Lab Site
```
Additional columns are ignored.

## ❓ FAQ
**Q: My CSV uses different header names.**  
A: The app already checks several variants (e.g., `Employment History` ⇒ **PI History**). If you need an exact custom mapping, open an issue or update `lookup_profile_sections` in `streamlit_app.py`.

**Q: Names don’t match from JSON to CSV.**  
A: The app guesses a PI key from the JSON filename (e.g., `RUnocic.json` → “Unocic”). Ensure the CSV `name` includes that last name. You can also rename JSON files for more reliable matching.

**Q: No results / zero similarity.**  
A: Try a longer description, add more JSON content, or rebuild vectors with the sidebar button.

## 📝 License
MIT (see `LICENSE` if added).
