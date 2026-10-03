import os
import streamlit as st
from pathlib import Path
from src.ingest import load_corpus
from src.retrieval import TfidfRetriever
from src.qa import answer_question

st.set_page_config(page_title="Document Q&A | TF-IDF RAG", page_icon="📄")
st.title("Document Q&A System")
st.caption("PDF retrieval with TF-IDF, cosine similarity, and grounded LLM answers.")

@st.cache_resource
def build_retriever():
    pdf_dir = Path(__file__).parent / "data" / "pdfs"
    return TfidfRetriever(load_corpus(pdf_dir, chunk_words=200))

try:
    retriever = build_retriever()
except Exception as exc:
    st.error(f"Could not index PDFs: {exc}")
    st.stop()

question = st.text_input("Ask a question about the indexed documents")
if st.button("Find answer", type="primary", disabled=not question.strip()):
    passages = retriever.search(question, top_k=3)
    st.subheader("Retrieved passages")
    for p in passages:
        with st.expander(f"{p['source']} · chunk {p['chunk_id']} · score {p['score']:.3f}"):
            st.write(p["text"])
    if not passages:
        st.warning("No relevant passages found.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.info("Set OPENAI_API_KEY to generate an answer. Retrieval is working.")
    else:
        with st.spinner("Generating an answer from retrieved passages..."):
            try:
                st.subheader("Answer")
                st.write(answer_question(question, passages))
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
