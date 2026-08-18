import os
import warnings
warnings.filterwarnings("ignore")  # إخفاء أي تحذيرات جانبيّة للشاشة

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. Load Vector Store
# ==========================================
print("📂 Loading Vector Database...")
db_dir = "./chroma_db_day2"
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_store = Chroma(
    persist_directory=db_dir,
    embedding_function=embeddings,
    collection_name="medical_rag_day2"
)

# ==========================================
# 2. Define System Prompt with Guardrails
# ==========================================
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

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# Initialize LLM using active Groq Model
llm = ChatGroq(
    model="openai/gpt-oss-120b",  # أو openai/gpt-oss-20b
    temperature=0.0
)

# ==========================================
# 3. Grounded Generation Pipeline Function
# ==========================================
def ask_rag_system(query: str, top_k: int = 4):
    print(f"\n❓ Query: '{query}'")

    docs = vector_store.similarity_search(query, k=top_k)

    context_str = ""
    for idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("document_id", "UNKNOWN")
        page = doc.metadata.get("page_number", "UNKNOWN")
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN")

        context_str += f"\n--- Source [{idx+1}]: [{doc_id} | p. {page} | {chunk_id}] ---\n"
        context_str += f"{doc.page_content}\n"

    formatted_prompt = prompt.format_messages(context=context_str, question=query)
    response = llm.invoke(formatted_prompt)

    print("\n🤖 AI Assistant Response:")
    print("--------------------------------------------------")
    print(response.content)
    print("--------------------------------------------------")

# ==========================================
# 4. Test Queries
# ==========================================
ask_rag_system("What are the recommendations for sun protection?")
ask_rag_system("How do I repair a car engine?")