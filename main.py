import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Step 1: Load summarization model offline
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("models/bart-large-cnn/")
    model = AutoModelForSeq2SeqLM.from_pretrained("models/bart-large-cnn/")
    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_summarizer()

# -----------------------------
# Step 2: Load sentence-transformer model offline
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_embedder():
    return SentenceTransformer("models/all-MiniLM-L6-v2")

embedder = load_embedder()

# -----------------------------
# Step 3: Functions
# -----------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def summarize_pdf(text, chunk_size=2000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summary = ""
    for chunk in chunks:
        s = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
        summary += s[0]['summary_text'] + " "
    return summary.strip()

summary_chunks = []
summary_embeddings = None

def prepare_summary_embeddings(summary):
    global summary_chunks, summary_embeddings
    summary_chunks = [summary[i:i+800] for i in range(0, len(summary), 800)]
    summary_embeddings = embedder.encode(summary_chunks, convert_to_tensor=True)

def answer_from_summary(question):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, summary_embeddings)[0]
    best_idx = scores.argmax().item()
    return summary_chunks[best_idx]

# -----------------------------
# Step 4: Streamlit UI
# -----------------------------
st.set_page_config(page_title="PDF Q&A Offline App", layout="wide")
st.title("📘 PDF Summarization & Q&A Offline App")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting PDF text..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    
    with st.spinner("Summarizing PDF..."):
        pdf_summary = summarize_pdf(pdf_text)
    
    prepare_summary_embeddings(pdf_summary)
    st.success("✅ PDF loaded and summarized successfully!")
    st.text_area("📄 PDF Summary (first 1000 chars)", pdf_summary[:1000], height=200)

    question = st.text_input("❓ Ask a question about the PDF:")
    if question:
        with st.spinner("Searching for answer..."):
            answer = answer_from_summary(question)
        st.write("📘 Answer:", answer)
