from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from src.config.config import GEMINI_API_KEY
import chromadb
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Re-initialize your environment
Settings.llm = GoogleGenAI(model='gemini-2.5-flash')
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Connect to your existing database
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("my_knowledge_base")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 3. Load the index from the database
index = VectorStoreIndex.from_vector_store(vector_store)

# 4. Create the query engine
query_engine = index.as_query_engine()

# 5. Ask a question!
response = query_engine.query("What are the key points in the AI Engineering book?")
print(response)