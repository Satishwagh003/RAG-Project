import streamlit as st
import uuid
import os
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
import os
import uuid
import pandas as pd
from pypdf import PdfReader
from pathlib import Path
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from langchain_huggingface import HuggingFaceEmbeddings
# FAISS index path
FAISS_INDEX_PATH = "faiss_index"

# -----------------------------
# Validation
# -----------------------------
def validate_env_vars():
    """Validate that all required environment variables are set"""
    required_vars = {
        "AZURE_OPENAI_API_KEY": AZURE_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_ENDPOINT,
        "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_DEPLOYMENT,
        "AZURE_OPENAI_MODEL": AZURE_MODEL,
        "AZURE_OPENAI_API_VERSION": AZURE_API_VERSION
    }
    
    missing = [key for key, value in required_vars.items() if not value]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# -----------------------------
# Helper functions
# -----------------------------

def get_pdf_text(pdf_doc) -> str:
    """Extract text from PDF file"""
    try:
        text = ""
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF {pdf_doc.name}: {str(e)}")
        return ""

def create_docs(pdf_files, unique_id: str) -> List[Document]:
    """Convert uploaded PDFs to Document objects"""
    docs = []
    for f in pdf_files:
        text = get_pdf_text(f)
        if text:  # Only add if text was extracted
            docs.append(Document(
                page_content=text,
                metadata={
                    "name": f.name,
                    "id": getattr(f, "file_id", None),
                    "type": getattr(f, "type", "pdf"),
                    "size": getattr(f, "size", None),
                    "unique_id": unique_id
                }
            ))
        else:
            st.warning(f"⚠️ Could not extract text from {f.name}")
    return docs

@st.cache_resource
def create_embeddings():
    """Create embeddings using SentenceTransformer (cached)"""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def save_faiss_index(docs: List[Document], embeddings) -> FAISS:
    """Create and save FAISS index locally"""
    if not docs:
        raise ValueError("No documents to index")
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Create directory if it doesn't exist
    Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

def load_faiss_index(embeddings, index_path: str = FAISS_INDEX_PATH) -> FAISS:
    """Load FAISS index from disk"""
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def similar_docs(query: str, k: int, embeddings) -> List[Tuple[Document, float]]:
    """
    Search FAISS index for similar documents.
    Returns documents sorted by similarity (best match first).
    Lower distance = better match.
    """
    vectorstore = load_faiss_index(embeddings)
    results = vectorstore.similarity_search_with_score(query, k=k)
    # Sort by distance (ascending) - lower distance means better match
    results_sorted = sorted(results, key=lambda x: x[1])
    return results_sorted

def get_llm():
    """Create Azure OpenAI LLM instance"""
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        model=AZURE_MODEL,
        temperature=0,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY
    )

def get_summary(doc: Document) -> str:
    """Summarize a document using Azure OpenAI"""
    try:
        llm = get_llm()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.invoke([doc])
        return summary.get("output_text", "Summary not available")
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def calculate_match_percentage(distance: float) -> float:
    """
    Convert FAISS distance to match percentage.
    FAISS uses L2 distance for most embeddings.
    Lower distance = better match.
    """
    # For L2 distance, typical range is 0 to ~2
    # We'll use an exponential decay to convert to percentage
    # This provides better differentiation between scores
    import math
    match_score = 100 * math.exp(-distance / 2)
    return max(0, min(100, match_score))

# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.set_page_config(
        page_title="Resume Screening Assistance",
        page_icon="💼",
        layout="wide"
    )
    
    st.title("HR - Resume Screening Assistance 💁")
    st.subheader("I can help you screen resumes faster!")
    
    # Validate environment variables
    try:
        validate_env_vars()
    except ValueError as e:
        st.error(str(e))
        st.info("Please create a `.env` file with the required Azure OpenAI credentials.")
        st.stop()
    
    # Initialize session state
    if 'unique_id' not in st.session_state:
        st.session_state['unique_id'] = ''
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        job_description = st.text_area(
            "Paste the JOB DESCRIPTION here...",
            height=200,
            placeholder="Enter the job requirements, skills, and qualifications..."
        )
    
    with col2:
        document_count = st.number_input(
            "No. of RESUMES to return",
            min_value=1,
            max_value=20,
            value=5
        )
    
    pdf_files = st.file_uploader(
        "Upload resumes (PDF only)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDF resumes for analysis"
    )
    
    # Analysis button
    if st.button("🔍 Help me with the analysis", type="primary"):
        if not job_description.strip():
            st.warning("Please enter a job description.")
            return
        
        if not pdf_files:
            st.warning("Please upload at least one resume.")
            return
        
        with st.spinner("Processing resumes..."):
            try:
                # Generate unique ID for this session
                st.session_state['unique_id'] = uuid.uuid4().hex
                
                # Create Document objects
                docs = create_docs(pdf_files, st.session_state['unique_id'])
                
                if not docs:
                    st.error("No text could be extracted from the uploaded PDFs.")
                    return
                
                st.info(f"✅ Successfully processed {len(docs)} resume(s)")
                
                # Create embeddings
                embeddings = create_embeddings()
                
                # Save FAISS index
                with st.spinner("Building search index..."):
                    save_faiss_index(docs, embeddings)
                
                # Fetch similar documents (already sorted best first)
                k = min(int(document_count), len(docs))
                with st.spinner("Analyzing resumes..."):
                    relevant_docs = similar_docs(job_description, k, embeddings)
                
                # Display results
                st.write("---")
                st.subheader("📊 Top Matching Resumes (Best to Worst)")
                
                for idx, (doc, distance) in enumerate(relevant_docs):
                    # Calculate match percentage
                    match_percentage = calculate_match_percentage(distance)
                    
                    # Color code based on match quality
                    if match_percentage >= 80:
                        badge = "🟢 Excellent Match"
                    elif match_percentage >= 60:
                        badge = "🟡 Good Match"
                    elif match_percentage >= 40:
                        badge = "🟠 Fair Match"
                    else:
                        badge = "🔴 Weak Match"
                    
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"### {idx + 1}. {doc.metadata.get('name', 'Unknown')}")
                        
                        with col2:
                            st.metric("Match Score", f"{match_percentage:.1f}%")
                        
                        with col3:
                            st.markdown(f"**{badge}**")
                        
                        with st.expander("📄 View Details"):
                            st.markdown("**Summary:**")
                            with st.spinner("Generating summary..."):
                                summary = get_summary(doc)
                            st.write(summary)
                            
                            st.markdown("**Metadata:**")
                            metadata_info = {
                                "File Name": doc.metadata.get('name', 'N/A'),
                                "File Size": f"{doc.metadata.get('size', 0) / 1024:.2f} KB" if doc.metadata.get('size') else "N/A",
                                "Distance Score": f"{distance:.4f}",
                                "Match Percentage": f"{match_percentage:.2f}%"
                            }
                            st.json(metadata_info)
                            
                            # Option to view full text
                            if st.checkbox(f"Show full resume text", key=f"show_text_{idx}"):
                                st.text_area(
                                    "Resume Content",
                                    doc.page_content[:2000] + ("..." if len(doc.page_content) > 2000 else ""),
                                    height=300,
                                    key=f"content_{idx}"
                                )
                        
                        st.write("")
                
                st.success("✅ Analysis complete! Hope this saved your time ❤️")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()