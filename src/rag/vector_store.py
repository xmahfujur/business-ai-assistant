import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.rag.loader import universal_doc_loader
from src.rag.chunker import chunker


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def build_vector_index():
    # 1. load and chunk my docs
    docs = universal_doc_loader('src/uploads')
    chunks = chunker(docs, chunk_size=512, chunk_overlap=50)
    
    # 2. Initialize chromadb client (store data in ./chromadb folder)
    db = chromadb.PersistentClient(path='./chroma_db')
    
    # 3. Create or get a collections
    chroma_collections = db.get_or_create_collection('my_knowledge_base')
    
    vector_store = ChromaVectorStore(chroma_collection = chroma_collections)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(chunks, storage_context=storage_context)
    
    print("Vector index built and saved to ./chroma_db")
    return index

if __name__ == "__main__":
    index = build_vector_index()   