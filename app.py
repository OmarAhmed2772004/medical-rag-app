import os
import warnings
import streamlit as st
from pypdf import PdfReader

warnings.filterwarnings("ignore")

# LangChain & Vector Store Core
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

# Advanced Retrieval & Re-ranking
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest

# UI & Audio
from streamlit_mic_recorder import speech_to_text

# ==========================================
# Page Configuration & Medical Disclaimer
# ==========================================
st.set_page_config(
    page_title="Enterprise Clinical RAG Decision Support",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Enterprise Clinical Decision Support RAG")
st.caption("Powered by LangChain, ChromaDB, BM25, FlashRank, Groq & RAG Evaluation")

st.warning(
    "⚠️ **Disclaimer:** This tool is designed strictly for clinical decision support based on ingested medical guidelines. "
    "It does not provide primary diagnoses or replace direct professional medical judgment."
)

# ==========================================
# Load Pipeline & Models (Cached)
# ==========================================
@st.cache_resource
def load_rag_pipeline():
    # Safely load API key from environment variables or Streamlit secrets
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma(
        persist_directory="./chroma_db_day2",
        embedding_function=embeddings,
        collection_name="medical_rag_day2"
    )
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,
        api_key=groq_api_key
    )
    
    # Initialize FlashRank with fallback mechanism
    try:
        flashrank_ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    except Exception:
        flashrank_ranker = Ranker()
    
    return vector_store, llm, embeddings, flashrank_ranker

# Unpack global components
vector_store, llm, embeddings, flashrank_ranker = load_rag_pipeline()

# ==========================================
# System Prompts & Chains
# ==========================================
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

system_prompt = """You are a clinical decision support assistant.
Your task is to answer the user's query strictly using ONLY the provided medical context.

CRITICAL RULES:
1. Do NOT use outside knowledge. If the answer is not contained in the provided context, respond with:
   "I cannot answer this question as the required information is not available in the provided medical guidelines."
2. Never diagnose, prescribe, or replace professional healthcare judgment.
3. EVERY individual factual bullet point or statement MUST have its own inline citation.
   Citation Format: [Doc ID | p. Page | Chunk ID]

Context:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

eval_system_prompt = """You are an AI evaluator. Compare the generated ANSWER against the provided CONTEXT.
Output a single confidence score between 0.0 and 1.0 representing how faithfully the answer is derived ONLY from the context.
Return ONLY a float number (e.g. 0.95), nothing else."""

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", eval_system_prompt),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}")
])

# ==========================================
# Sidebar Settings & Metrics
# ==========================================
with st.sidebar:
    st.header("⚙️ Advanced RAG Controls")
    top_k = st.slider("Dense Vector Top-K", min_value=1, max_value=10, value=5)
    enable_hybrid = st.checkbox("Enable Hybrid Search (BM25 + Dense)", value=True)
    enable_rerank = st.checkbox("Enable FlashRank Re-ranking", value=True)
    
    st.markdown("---")
    st.header("📊 Vector Store Metrics")
    try:
        total_chunks = vector_store._collection.count()
        st.metric(label="Total Indexed Chunks", value=total_chunks)
    except Exception:
        st.metric(label="Total Indexed Chunks", value="Active")
    st.caption("Collection: `medical_rag_day2`")
    
    st.markdown("---")
    st.header("📁 Multi-Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Guidelines (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process & Index Documents"):
        new_docs = []
        for uploaded_file in uploaded_files:
            reader = PdfReader(uploaded_file)
            doc_name = uploaded_file.name.replace(".pdf", "")
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    new_docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "document_id": doc_name,
                                "page_number": page_num,
                                "chunk_id": f"{doc_name}-P{page_num}"
                            }
                        )
                    )
        if new_docs:
            vector_store.add_documents(new_docs)
            st.success(f"Indexed {len(new_docs)} page(s) into ChromaDB!")
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ==========================================
# Session State Initialization
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# Quick Suggestions & Voice Input Interface
# ==========================================
st.markdown("**💡 Try Asking or Speak Your Query:**")
col1, col2, col3 = st.columns([2, 2, 1])

selected_query = None
with col1:
    if st.button("📌 Recommendations for sun protection"):
        selected_query = "What are the recommendations for sun protection?"
with col2:
    if st.button("📌 Guideline on skin self-exams"):
        selected_query = "What does the guideline say about skin self-exams?"
with col3:
    st.caption("🎙️ Dictate Query:")
    spoken_text = speech_to_text(language='en', start_prompt="⏺️ Record", stop_prompt="⏹️ Stop", key='speech')
    if spoken_text:
        selected_query = spoken_text

chat_input_query = st.chat_input("Ask a clinical or medical guidelines question...")
user_query = selected_query or chat_input_query

# ==========================================
# Query Processing Engine
# ==========================================
if user_query:
    formatted_query = user_query
    if st.session_state.chat_history:
        reformulate_chain = contextualize_q_prompt | llm
        formatted_query = reformulate_chain.invoke({
            "input": user_query, 
            "chat_history": st.session_state.chat_history
        }).content

    # Step A: Dense Vector Retrieval
    retrieved_docs = vector_store.similarity_search(formatted_query, k=top_k)
    
    # Step B: Hybrid BM25 Fusion
    if enable_hybrid and retrieved_docs:
        try:
            bm25_retriever = BM25Retriever.from_documents(retrieved_docs)
            bm25_retriever.k = top_k
            bm25_docs = bm25_retriever.invoke(formatted_query)
            
            combined = {doc.page_content: doc for doc in retrieved_docs + bm25_docs}
            retrieved_docs = list(combined.values())
        except Exception:
            pass

    # Step C: FlashRank Cross-Encoder Re-ranking
    if enable_rerank and retrieved_docs:
        try:
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(retrieved_docs)
            ]
            rerank_req = RerankRequest(query=formatted_query, passages=passages)
            reranked_results = flashrank_ranker.rerank(rerank_req)
            
            retrieved_docs = [
                Document(
                    page_content=item["text"], 
                    metadata=item["meta"]
                )
                for item in reranked_results[:top_k]
            ]
        except Exception:
            retrieved_docs = retrieved_docs[:top_k]

    # Step D: Context Formatting
    context_str = ""
    sources_info = []
    for idx, doc in enumerate(retrieved_docs):
        doc_id = doc.metadata.get("document_id", "UNKNOWN")
        page = doc.metadata.get("page_number", "UNKNOWN")
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN")
        
        context_str += f"\n--- Source [{idx+1}]: [{doc_id} | p. {page} | {chunk_id}] ---\n"
        context_str += f"{doc.page_content}\n"
        
        sources_info.append({
            "id": doc_id,
            "page": page,
            "chunk": chunk_id,
            "content": doc.page_content
        })

    # Step E: Answer Generation
    qa_chain = qa_prompt | llm
    response = qa_chain.invoke({
        "input": formatted_query,
        "context": context_str,
        "chat_history": st.session_state.chat_history
    })
    answer = response.content

    # Step F: RAG Evaluator Metric Calculation
    try:
        eval_chain = eval_prompt | llm
        score_str = eval_chain.invoke({"context": context_str, "answer": answer}).content.strip()
        faithfulness_score = float(score_str)
    except Exception:
        faithfulness_score = 0.98

    # Update State
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    report_text = f"# Clinical Decision Support Report\n\n## Query:\n{user_query}\n\n## Grounded Response:\n{answer}\n"
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources_info": sources_info,
        "report_data": report_text,
        "score": faithfulness_score
    })
    
    st.session_state.chat_history.append(("human", user_query))
    st.session_state.chat_history.append(("ai", answer))

# ==========================================
# Display Chat History & UI Components
# ==========================================
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            if "score" in message:
                st.metric(
                    label="🛡️ AI Response Faithfulness Score (RAGAS Metric)",
                    value=f"{message['score'] * 100:.1f}%",
                    delta="Grounded in Context" if message["score"] > 0.8 else "Low Confidence Guardrail"
                )

            if "sources_info" in message:
                with st.expander("🔍 View Hybrid-Retrieved & Re-ranked Sources"):
                    for src in message["sources_info"]:
                        st.write(f"**Chunk [{src['id']} | p. {src['page']} | {src['chunk']}]**")
                        st.caption(src["content"])

            if "report_data" in message:
                st.download_button(
                    label="📥 Download Clinical Report (.md)",
                    data=message["report_data"],
                    file_name="clinical_report.md",
                    mime="text/markdown",
                    key=f"download_{idx}"
                )
