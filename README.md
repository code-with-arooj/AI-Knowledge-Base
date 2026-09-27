# 📚 Personal AI Knowledge Base

An advanced **Retrieval-Augmented Generation (RAG)** system that allows users to ask questions from PDF documents and get accurate, source-cited answers.

Built completely with **free and local tools** — no paid APIs required.

---

## 🚀 Features

- **Hybrid Search** (BM25 + Vector Search) for better retrieval quality
- Local LLM using **Ollama** (`phi3:mini`)
- Streaming responses for better user experience
- Source citations with page numbers
- Optimized to run on **8GB RAM** laptops
- Clean and simple Streamlit UI
- Support for multiple PDFs

---

## 🛠️ Tech Stack

| Component              | Technology                      |
|------------------------|---------------------------------|
| Framework              | Streamlit                       |
| Orchestration          | LangChain                       |
| Vector Database        | ChromaDB                        |
| Embeddings             | sentence-transformers (all-MiniLM-L6-v2) |
| Keyword Search         | BM25                            |
| LLM                    | Ollama (phi3:mini)              |
| Document Loader        | PyPDF                           |

---

## 📂 Project Structure

```bash
AI_Knowledge_Base/
│
├── documents/                 # Place your PDF files here
├── chroma_db/                 # Vector database (auto-generated)
├── app.py                     # Main application
├── requirements.txt
└── README.md

---

⚙️ How to Run
1. Clone the repository
Bashgit clone https://github.com/YourUsername/AI-Knowledge-Base.git
cd AI-Knowledge-Base
2. Create virtual environment
Bashpython -m venv venv
venv\Scripts\activate          # Windows
3. Install dependencies
Bashpip install -r requirements.txt
4. Install Ollama & Model

Download Ollama from https://ollama.com
Run the model:

Bashollama run phi3:mini
5. Add your PDFs
Put PDF files inside the documents/ folder.
6. Run the app
Bashstreamlit run app.py

🧠 How It Works

PDFs are loaded and split into chunks
Chunks are converted into embeddings and stored in ChromaDB
User question is searched using Hybrid Search (BM25 + Vector)
Top relevant chunks are passed to the local LLM
LLM generates an answer strictly based on the retrieved context
Sources with page numbers are shown


⚔️ Challenges Faced & Solutions

## ⚔️ Challenges Faced & Solutions

| Challenge | Solution |
|---------|----------|
| PowerShell script execution policy error | Used `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| High RAM usage with large PDFs | Used smaller chunk size (500) + lightweight embedding model (`all-MiniLM-L6-v2`) |
| Hallucinations by LLM | Applied strict prompt engineering so the model only answers from the given context |
| Slow response on 8GB RAM laptop | Used lightweight model `phi3:mini` + streaming responses |
| Duplicate retrieval results | Added custom deduplication logic while combining BM25 and Vector results |
| EnsembleRetriever import issues due to LangChain version conflicts | Implemented custom Hybrid Search (BM25 + Vector) manually for better stability |
| Ollama command not recognized in some terminals | Ran Ollama in a fresh system terminal instead of inside the virtual environment |

📈 Future Improvements

Add re-ranking for better precision
Support for more document types (DOCX, TXT, etc.)
Evaluation metrics (Faithfulness, Answer Relevance)
Docker support for easy deployment
User authentication and multi-user support


💡 Why This Project?
This project demonstrates practical understanding of:

Retrieval-Augmented Generation (RAG)
Hybrid Search techniques
Local LLM integration
Performance optimization on limited hardware
End-to-end AI application development

Built with a strong focus on real-world constraints (low RAM, free tools only).

🙌 Author
Arooj

Aspiring AI / Machine Learning Engineer
