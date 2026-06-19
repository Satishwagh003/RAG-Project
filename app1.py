import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from docx import Document
from pydantic import SecretStr
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------- SIMPLE FALLBACK QA CHAIN ----------------
class SimpleStuffChain:
    """
    Minimal replacement for deprecated load_qa_chain.
    Builds a prompt with context + question and calls the LLM explicitly.
    Works across LangChain versions and avoids callable-model issues.
    """

    def __init__(self, llm, prompt_template: str):
        self.llm = llm
        self.prompt_template = prompt_template

    def invoke(self, inputs: dict):
        context = inputs.get("context", "")

        # Combine retrieved docs
        if isinstance(context, list):
            context_text = "\n\n".join(
                [getattr(d, "page_content", str(d)) for d in context]
            )
        else:
            context_text = str(context)

        question = inputs.get("question", "")

        prompt_text = self.prompt_template.format(
            context=context_text,
            question=question
        )

        # Call AzureChatOpenAI safely
        result = self.llm.invoke(prompt_text)

        # Normalize Azure responses
        if hasattr(result, "content"):
            return result.content

        return str(result)


# ---------------- ENV ----------------
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")

FAISS_INDEX_PATH = "faiss_index"


# ---------------- PDF TEXT ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


# ---------------- EXCEL TEXT ----------------
def get_excel_text(excel_docs):
    chunks = []
    for excel in excel_docs:
        df = pd.read_excel(excel, sheet_name=None, dtype=str)
        for sheet in df.values():
            for _, row in sheet.iterrows():
                row_values = row.dropna().astype(str)
                chunks.append(" | ".join(row_values.tolist()))
    return chunks


# ---------------- TXT / DOCX ----------------
def get_text_from_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    return ""


# ---------------- CHUNKING ----------------
def get_text_chunks(text_chunks):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.split_text("\n".join(text_chunks))


# ---------------- VECTOR STORE ----------------
def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local(FAISS_INDEX_PATH)

    st.success("✅ FAISS index created successfully! You can now ask questions.")


# ---------------- QA CHAIN ----------------
def get_conversational_chain():

    prompt_template = """
    You are an AI assistant tasked with answering questions based only on the provided context.
    Also give all the Follow Up Questions linked to the asked question from the provided context.
    Use the provided context to answer the question with the highest possible accuracy.
    Do not use external knowledge—only rely on the given context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=api_version,
        azure_deployment=deployment_name,   # correct param
        api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,
        temperature=0
    )

    return SimpleStuffChain(llm, prompt_template)


# ---------------- STREAMLIT APP ----------------
def main():

    st.set_page_config(
        page_title="QnA ChatBot FAQs from Documents",
        page_icon="📘",
        layout="wide"
    )

    st.header("QnA ChatBot FAQs from Documents 💬")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.title("Upload & Process Files")

        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        excel_docs = st.file_uploader("Upload Excel Files", type=["xls", "xlsx"], accept_multiple_files=True)
        txt_docs = st.file_uploader("Upload TXT / DOCX Files", type=["txt", "docx"], accept_multiple_files=True)

        if st.button("Process Documents"):
            raw_text = []

            if pdf_docs:
                raw_text.append(get_pdf_text(pdf_docs))

            if excel_docs:
                raw_text.extend(get_excel_text(excel_docs))

            if txt_docs:
                for doc in txt_docs:
                    raw_text.append(get_text_from_file(doc))

            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)


    # ---------- CLEAR CHAT ----------
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


    # ---------- QUESTION ----------
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question here 👇")


    if user_question:

        if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
            st.error("❌ FAISS index not found. Please upload and process documents first.")

        else:
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            docs = vector_store.similarity_search(user_question, k=3)

            chain = get_conversational_chain()

            response = chain.invoke({
                "context": docs,
                "question": user_question
            })

            st.session_state.chat_history.append((user_question, response))


    # ---------- CHAT HISTORY ----------
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")

        for q, a in st.session_state.chat_history:
            st.markdown(f"**🧑 You:** {q}")
            st.markdown(f"**🤖 AI:** {a}")


if __name__ == "__main__":
    main()
