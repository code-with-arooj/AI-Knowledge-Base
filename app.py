import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os

st.set_page_config(page_title="AI Knowledge Base", page_icon="📚", layout="wide")

st.title("📚 Personal AI Knowledge Base")
st.caption("Advanced RAG • Hybrid Search • Local & Free")

with st.sidebar:
    st.header("System Info")
    st.write("• Hybrid Search (BM25 + Vector)")
    st.write("• Model: phi3:mini")
    st.write("• Vector DB: ChromaDB")
    st.divider()
    st.write("Put PDF files in the `documents` folder")

@st.cache_resource
def load_retrievers():
    documents = []
    pdf_files = [f for f in os.listdir("documents") if f.endswith(".pdf")]
    
    if not pdf_files:
        st.error("No PDF found in documents folder!")
        st.stop()
    
    for file in pdf_files:
        loader = PyPDFLoader(f"documents/{file}")
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Vector Retriever
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    return vector_retriever, bm25_retriever, len(pdf_files), len(chunks)

with st.spinner("Loading Hybrid Knowledge Base..."):
    vector_retriever, bm25_retriever, num_pdfs, num_chunks = load_retrievers()

st.success(f"Loaded {num_pdfs} PDF(s) • {num_chunks} chunks • Hybrid Search Active")

llm = OllamaLLM(model="phi3:mini")

template = """You are a helpful assistant. 
Answer the question using ONLY the information given in the Context below. 
Do not use any outside knowledge. 
If the answer is not clearly present in the Context, say: "I don't have enough information from the document."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def hybrid_retrieve(query):
    # Get results from both retrievers
    vector_docs = vector_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)
    
    # Combine and remove duplicates
    all_docs = []
    seen = set()
    
    for doc in vector_docs + bm25_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            all_docs.append(doc)
    
    return all_docs[:5]   # top 5 unique results

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Example Questions
st.subheader("Try these questions:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("What is a program?"):
        st.session_state.query = "What is a program?"
with col2:
    if st.button("What is debugging?"):
        st.session_state.query = "What is debugging?"
with col3:
    if st.button("What is the trap door effect?"):
        st.session_state.query = "What is the trap door effect?"

query = st.text_input("Enter your question:", value=st.session_state.get("query", ""))

if query:
    st.subheader("Answer")
    with st.spinner("Generating answer with Hybrid Search..."):
        docs = hybrid_retrieve(query)
        context = format_docs(docs)
        
        chain = prompt | llm | StrOutputParser()
        
        answer_placeholder = st.empty()
        full_answer = ""
        for chunk in chain.stream({"context": context, "question": query}):
            full_answer += chunk
            answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)

    # Sources
    st.subheader("Sources (Hybrid Retrieval)")
    for i, doc in enumerate(docs, 1):
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        with st.expander(f"Source {i}  |  {source_name}  |  Page {page}"):
            st.write(doc.page_content)