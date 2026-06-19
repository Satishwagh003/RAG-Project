import streamlit as st
import uuid
import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from pypdf import PdfReader
import pandas as pd
import math
from pydantic import SecretStr

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import AzureChatOpenAI

# New import path for summarize chain (LangChain >= 0.2)
try:
    from langchain.chains.summarize import load_summarize_chain  # type: ignore
except Exception:
    # Summarize chain isn't available in this environment; fall back to excerpt-based summary
    load_summarize_chain = None


# ==========================
# Load ENV variables
# ==========================

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

FAISS_INDEX_PATH = "faiss_index"


# ==========================
# Validate ENV
# ==========================

def validate_env_vars():
    required = {
        "AZURE_OPENAI_API_KEY": AZURE_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_ENDPOINT,
        "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_DEPLOYMENT,
        "AZURE_OPENAI_MODEL": AZURE_MODEL,
        "AZURE_OPENAI_API_VERSION": AZURE_API_VERSION
    }

    missing = [k for k, v in required.items() if not v]

    if missing:
        raise ValueError(
            "❌ Missing environment variables:\n"
            + "\n".join(missing)
            + "\n\nAdd them in your .env file."
        )


# ==========================
# PDF Reader
# ==========================

def get_pdf_text(pdf_doc) -> str:
    try:
        text = ""
        reader = PdfReader(pdf_doc)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception as e:
        st.error(f"Error reading {pdf_doc.name}: {e}")
        return ""


def create_docs(pdf_files, unique_id: str) -> List[Document]:
    docs = []

    for f in pdf_files:
        text = get_pdf_text(f)

        if not text:
            st.warning(f"⚠️ Could not extract text from {f.name}")
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "name": f.name,
                    "size": getattr(f, "size", None),
                    "unique_id": unique_id
                }
            )
        )

    return docs


# ==========================
# Embeddings
# ==========================

@st.cache_resource
def create_embeddings():
    return SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


# ==========================
# FAISS Index
# ==========================

def save_faiss_index(docs, embeddings):
    Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)

    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_INDEX_PATH)

    return vs


def load_faiss_index(embeddings):
    if not Path(FAISS_INDEX_PATH).exists():
        raise FileNotFoundError("⚠️ No FAISS index found. Please upload resumes first.")

    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def similar_docs(query: str, k: int, embeddings):
    vs = load_faiss_index(embeddings)

    results = vs.similarity_search_with_score(query, k=k)

    # lower score = better match
    return sorted(results, key=lambda x: x[1])


# ==========================
# Azure OpenAI LLM
# ==========================

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        model=AZURE_MODEL,
        temperature=0,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=SecretStr(AZURE_API_KEY) if AZURE_API_KEY else None
    )


# ==========================
# Resume Summary
# ==========================

def get_summary(doc: Document):
    try:
        llm = get_llm()

        if load_summarize_chain:
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce"
            )
            out = chain.invoke([doc])
            return out.get("output_text", "Summary not available")
        else:
            # Fallback: return a concise excerpt when summarize chain is unavailable
            excerpt = doc.page_content.strip()
            return excerpt[:2000] + ("..." if len(excerpt) > 2000 else "")

    except Exception as e:
        return f"⚠️ Error generating summary: {e}"


# ==========================
# Match Score
# ==========================

def calculate_match_percentage(distance: float):
    score = 100 * math.exp(-distance / 2)
    return max(0, min(100, score))


# ==========================
# Streamlit UI
# ==========================

def main():

    st.set_page_config(
        page_title="Resume Screening Assistant",
        page_icon="💼",
        layout="wide"
    )

    st.title("💼 HR Resume Screening Assistant")
    st.caption("Find best matching resumes for a job description")

    try:
        validate_env_vars()
    except Exception as e:
        st.error(str(e))
        st.stop()

    if "unique_id" not in st.session_state:
        st.session_state.unique_id = ""

    col1, col2 = st.columns([3, 1])

    with col1:
        job_description = st.text_area(
            "Paste Job Description",
            height=200
        )

    with col2:
        document_count = st.number_input(
            "Resumes to return",
            min_value=1,
            max_value=20,
            value=5
        )

    pdf_files = st.file_uploader(
        "Upload Resumes (PDF only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🔍 Analyze Resumes", type="primary"):

        if not job_description.strip():
            st.warning("Please enter a job description")
            return

        if not pdf_files:
            st.warning("Please upload resumes")
            return

        with st.spinner("Processing resumes..."):

            st.session_state.unique_id = uuid.uuid4().hex

            docs = create_docs(pdf_files, st.session_state.unique_id)

            if not docs:
                st.error("No text extracted from resumes")
                return

            embeddings = create_embeddings()

            save_faiss_index(docs, embeddings)

            k = min(int(document_count), len(docs))

            relevant_docs = similar_docs(job_description, k, embeddings)

        st.divider()
        st.subheader("📊 Best Matching Resumes")

        for idx, (doc, distance) in enumerate(relevant_docs):

            score = calculate_match_percentage(distance)

            if score >= 80:
                badge = "🟢 Excellent Match"
            elif score >= 60:
                badge = "🟡 Good Match"
            elif score >= 40:
                badge = "🟠 Fair Match"
            else:
                badge = "🔴 Weak Match"

            with st.container():

                c1, c2, c3 = st.columns([2, 1, 1])

                with c1:
                    st.markdown(f"### {idx+1}. {doc.metadata.get('name','Resume')}")

                with c2:
                    st.metric("Match Score", f"{score:.1f}%")

                with c3:
                    st.write(badge)

                with st.expander("📄 View Summary"):

                    st.markdown("**Summary:**")

                    with st.spinner("Summarizing..."):
                        summary = get_summary(doc)

                    st.write(summary)

                    st.json({
                        "File Name": doc.metadata.get("name"),
                        "Distance Score": f"{distance:.4f}",
                        "Match %": f"{score:.2f}%"
                    })

                    if st.checkbox(f"Show Full Text", key=f"t{idx}"):
                        st.text_area(
                            "Resume Content",
                            doc.page_content[:2000] +
                            ("..." if len(doc.page_content) > 2000 else ""),
                            height=300
                        )

        st.success("✅ Resume analysis completed")


if __name__ == "__main__":
    main()
