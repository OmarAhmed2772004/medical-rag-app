import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 0. Clean up previous vector store (Prevents Duplicates)
# ==========================================
db_dir = "./chroma_db_day2"
if os.path.exists(db_dir):
    shutil.rmtree(db_dir)
    print("🧹 Previous database cleared to prevent duplicate results.")

# ==========================================
# 1. Load PDF Document (PDF Ingestion)
# ==========================================
print("\n📂 Loading PDF file...")
pdf_path = "data/medical_guideline.pdf"
loader = PyPDFLoader(pdf_path)
raw_documents = loader.load()

# Attach metadata to raw pages
for idx, doc in enumerate(raw_documents):
    doc.metadata["document_id"] = "CLIN-SC-2026-001"
    doc.metadata["page_number"] = idx + 1

print(f"✅ Successfully loaded {len(raw_documents)} pages.")

# ==========================================
# 2. Text Chunking (Small Chunk Size)
# ==========================================
print("\n✂️ Splitting text into smaller chunks (Size: 500, Overlap: 75)...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    length_function=len
)

chunks = text_splitter.split_documents(raw_documents)

# Assign unique chunk IDs
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"{chunk.metadata['document_id']}-CH-{i+1:03d}"

print(f"✅ Document split into {len(chunks)} chunks.")

# ==========================================
# 3. Vector Embeddings & Indexing
# ==========================================
print("\n🧠 Generating embeddings and storing in ChromaDB...")

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="medical_rag_day2",
    persist_directory=db_dir,
    collection_metadata={"hnsw:space": "cosine"}
)

print("✅ Embeddings saved to ChromaDB successfully.")

# ==========================================
# 4. Top-K Retrieval Evaluation
# ==========================================
query = "What are the sun protection recommendations?"
print(f"\n🔍 Searching for query: '{query}'")

for k_val in [3, 5]:
    print(f"\n================ Top-K = {k_val} ================")
    results = vector_store.similarity_search(query, k=k_val)
    
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(f"📌 Page: {doc.metadata.get('page_number')} | Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"📝 Text: {doc.page_content[:150]}...")