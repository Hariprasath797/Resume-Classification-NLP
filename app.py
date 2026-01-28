# ===============================
# Resume Classification App
# ===============================

import os
import io
import re
import tempfile
import pickle
import pandas as pd
import streamlit as st

# NLP libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# File parsing
import pdfplumber
import docx2txt

# ===============================
# Paths (VERY IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "deployment", "modelDT.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "deployment", "vector.pkl")
SKILLS_CSV_PATH = os.path.join(BASE_DIR, "deployment", "skills.csv")

# ===============================
# Ensure NLTK data
# ===============================
def ensure_nltk_data():
    needed = ["wordnet", "omw-1.4", "punkt", "stopwords"]
    for pkg in needed:
        try:
            if pkg == "punkt":
                nltk.data.find(f"tokenizers/{pkg}")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)

ensure_nltk_data()

STOPWORDS = set(nltk.corpus.stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"\w+")

# ===============================
# UI
# ===============================
st.title("RESUME CLASSIFICATION")
st.markdown("<style>h1{color: Purple;}</style>", unsafe_allow_html=True)
st.subheader("Welcome to Resume Classification App")

# ===============================
# Skills (DISABLED SAFELY)
# ===============================
def extract_skills(resume_text: str):
    return []

# ===============================
# Text extraction helpers
# ===============================
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            ptext = page.extract_text()
            if ptext:
                text.append(ptext)
    return "\n".join(text)

def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(docx_bytes)
        tmp_path = tmp.name
    try:
        return docx2txt.process(tmp_path) or ""
    finally:
        os.remove(tmp_path)

def get_text_from_uploaded(uploaded_file) -> str:
    uploaded_file.seek(0)
    data = uploaded_file.read()

    mime = uploaded_file.type or ""
    name = uploaded_file.name.lower()

    if "word" in mime or name.endswith(".docx"):
        return extract_text_from_docx_bytes(data)
    elif "pdf" in mime or name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    else:
        return ""

# ===============================
# Preprocessing
# ===============================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]+", "", text)

    tokens = tokenizer.tokenize(text)
    filtered = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemma_words)

# ===============================
# Load model & vectorizer
# ===============================
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
except Exception as e:
    st.error(f"Failed to load model/vectorizer: {e}")
    st.stop()

# ===============================
# File upload & prediction
# ===============================
upload_files = st.file_uploader(
    "Upload Your Resumes",
    type=["docx", "pdf"],
    accept_multiple_files=True
)

results = []

if upload_files:
    for uploaded in upload_files:
        try:
            text = get_text_from_uploaded(uploaded)
            cleaned = preprocess(text)
            prediction = model.predict(vectorizer.transform([cleaned]))[0]

            results.append({
                "Uploaded File": uploaded.name,
                "Predicted Profile": prediction
            })
        except Exception as e:
            results.append({
                "Uploaded File": uploaded.name,
                "Predicted Profile": f"Error: {e}"
            })

# ===============================
# Display results
# ===============================
if results:
    df = pd.DataFrame(results)
    st.table(df)

    select = ["PeopleSoft", "SQL Developer", "React JS Developer", "Workday"]
    st.subheader("Select as per Requirement")

    option = st.selectbox("Fields", select)

    if option:
        st.table(df[df["Predicted Profile"] == option])
else:
    st.info("Upload one or more resumes to see predictions.")
