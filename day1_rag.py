import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. تحميل ملف الـ PDF (PDF Ingestion)
# ==========================================
print("📂 جاري تحميل ملف الـ PDF...")
pdf_path = "data/medical_guideline.pdf"

# استخدام PyPDFLoader لقراءة صفحات الملف
loader = PyPDFLoader(pdf_path)
raw_documents = loader.load()

# إضافة البيانات الوصفية (Metadata) لتعزيز موثوقية المصادر
for idx, doc in enumerate(raw_documents):
    doc.metadata["document_id"] = "CLIN-SC-2026-001"
    doc.metadata["title"] = "Medical Guideline"
    doc.metadata["page_number"] = idx + 1

print(f"✅ تم تحميل {len(raw_documents)} صفحة بنجاح!")

# ==========================================
# 2. تقسيم النصوص إلى أجزاء (Text Chunking)
# ==========================================
print("\n✂️ جاري تقسيم النصوص إلى أجزاء صغيرة (Chunks)...")

# إعداد مقسّم النصوص مع حجم chunk=850 وتداخل overlap=150
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=850,
    chunk_overlap=150,
    length_function=len
)

chunks = text_splitter.split_documents(raw_documents)

# إعطاء كل جزء معرف فريد (Chunk ID) لتسهيل التتبع
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"{chunk.metadata['document_id']}-CH-{i+1:03d}"

print(f"✅ تم تقطيع المستند إلى {len(chunks)} جزء (Chunk).")

# ==========================================
# 3. التحويل لمتجهات وحفظها (Embeddings & VectorDB)
# ==========================================
print("\n🧠 جاري تحويل النصوص إلى متجهات وحفظها في قاعدة البيانات...")

# استخدام نموذج FastEmbed مجاني لتوليد المتجهات بدون الحاجة لمفاتيح API
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# إنشاء قاعدة بيانات متجهات Chroma وحفظ البيانات محلية بدقة Cosine Similarity
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="medical_rag_day1",
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)

print("✅ تم حفظ المتجهات في ChromaDB بنجاح!")

# ==========================================
# 4. اختبار الاسترجاع (Top-K Retrieval Test)
# ==========================================
print("\n🔍 جاري اختبار استرجاع المعلومات (Top-K Retrieval)...")

# تجربة بحث عن سؤال طبي داخل المستند مع استرجاع أعلى 4 نتائج (K=4)
query = "What are the sun protection recommendations?"
retrieved_docs = vector_store.similarity_search(query, k=4)

print(f"\n--- نتائج البحث عن: '{query}' ---")
for i, doc in enumerate(retrieved_docs):
    print(f"\n النتيجة [{i+1}]:")
    print(f"📌 المصدر: {doc.metadata.get('document_id')} | صفحة: {doc.metadata.get('page_number')} | رمز الجزء: {doc.metadata.get('chunk_id')}")
    print(f"📝 النص: {doc.page_content[:200]}...")