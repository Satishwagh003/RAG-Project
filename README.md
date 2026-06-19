# RAG-Project
# 🚀 DocuMind AI

### 🧠 Enterprise Document Intelligence Platform using RAG, FAISS & Azure OpenAI

Transform unstructured documents into an intelligent knowledge base with Retrieval-Augmented Generation (RAG).

---

## 🌟 Key Highlights

✅ Multi-Format Document Support (PDF, Excel, DOCX, TXT)

✅ Retrieval-Augmented Generation (RAG) Architecture

✅ Azure OpenAI Integration

✅ Semantic Search using FAISS Vector Database

✅ Sentence Transformer Embeddings

✅ Interactive Streamlit Chat Interface

✅ Context-Aware Question Answering

✅ Follow-Up Question Generation

✅ Enterprise Knowledge Management Solution

---

## 🎯 Business Problem

Organizations store valuable information across hundreds of documents.

Finding specific information manually can be:

❌ Time Consuming

❌ Inefficient

❌ Error Prone

DocuMind AI solves this problem by enabling users to ask questions in natural language and instantly receive accurate answers from uploaded documents.

---

## 🏗️ System Architecture

📂 Upload Documents

⬇️

📄 Text Extraction

(PDF / Excel / DOCX / TXT)

⬇️

✂️ Text Chunking

(RecursiveCharacterTextSplitter)

⬇️

🔢 Embedding Generation

(all-MiniLM-L6-v2)

⬇️

🗄️ FAISS Vector Database

⬇️

🔍 Semantic Search

⬇️

🤖 Azure OpenAI

⬇️

💡 Intelligent Answers

---

## ⚙️ Technology Stack

### 🎨 Frontend

* Streamlit

### 🖥️ Backend

* Python

### 🤖 AI & LLM

* Azure OpenAI
* LangChain

### 🔍 Embedding Models

* Sentence Transformers
* all-MiniLM-L6-v2

### 🗄️ Vector Database

* FAISS

### 📊 Data Processing

* Pandas
* PyPDF2
* python-docx

### 🔐 Configuration

* python-dotenv

---

## 📌 Core Features

### 📂 Document Upload

Upload multiple documents simultaneously.

Supported formats:

* PDF
* Excel (.xls, .xlsx)
* DOCX
* TXT

### 📑 Intelligent Text Processing

* Text Extraction
* Data Cleaning
* Context Preservation
* Recursive Chunking

### 🔍 Semantic Retrieval

Retrieve the most relevant information based on meaning instead of keyword matching.

### 🤖 AI-Powered Question Answering

Generate highly accurate answers using Azure OpenAI and retrieved context.

### 💬 Conversational Interface

* Chat History
* Follow-Up Questions
* Interactive User Experience

---

## 🔄 Project Workflow

### 1️⃣ Upload Documents

Users upload one or more documents.

### 2️⃣ Extract Content

Text is extracted from all supported file formats.

### 3️⃣ Create Chunks

Large documents are divided into smaller overlapping chunks.

### 4️⃣ Generate Embeddings

Chunks are converted into vector representations.

### 5️⃣ Store in FAISS

Vectors are indexed for fast retrieval.

### 6️⃣ Ask Questions

Users enter natural language questions.

### 7️⃣ Retrieve Relevant Context

Most relevant document chunks are fetched.

### 8️⃣ Generate Answer

Azure OpenAI generates context-aware responses.

### 9️⃣ Display Results

Answers and follow-up questions are presented to the user.

---

## 📈 Real-World Applications

🏢 Enterprise Knowledge Base

📚 Training Material Assistant

📋 SOP Search System

👨‍💼 HR Policy Assistant

🎓 Educational Content Search

📞 Customer Support Knowledge Base

📑 Compliance Document Retrieval

---

## 🔥 Project Achievements

✔ Built an end-to-end RAG pipeline.

✔ Reduced dependency on manual document searching.

✔ Implemented semantic search using FAISS.

✔ Integrated Azure OpenAI for contextual responses.

✔ Enabled multi-document and multi-format ingestion.

✔ Designed scalable document processing architecture.

✔ Developed production-ready Streamlit interface.

---

## 🚀 Future Enhancements

🔹 Conversation Memory

🔹 OCR for Scanned PDFs

🔹 User Authentication

🔹 Source Citation Display

🔹 Hybrid Search

🔹 Multi-Language Support

🔹 Analytics Dashboard

🔹 Document Summarization

---

## 📸 Application Screenshots

Add screenshots here

### 🏠 Home Page

<img width="100%" alt="Home Page" src="images/home.png">

### 💬 Chat Interface

<img width="100%" alt="Chat Interface" src="images/chat.png">

---

