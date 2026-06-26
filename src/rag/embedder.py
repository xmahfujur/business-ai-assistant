from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from src.rag.loader import universal_doc_loader
from src.rag.chunker import chunker

def main():
    print(".........Document Loading.........")
    docs = universal_doc_loader('src/uploads')
    if not docs:
        print('No documents found')
        return
    
    print('--------chunking documnets---------')
    chunks = chunker(docs, chunk_size=512, chunk_overlap=50)
    print(f"Successfully split {len(docs)} pages into {len(chunks)} text chunks.")
    
    print('----Initializing Local Embedding Model----')
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    print('----Huggingface BGE Model Configure Successfully----')
    
    if chunks:
        all_text = [chunk.text for chunk in chunks]
        
        all_embeddings = embed_model.get_text_embedding_batch(all_text, show_progress = True)
        
        print('/n====SUCCESS====')
        
        # print(f"Total embeddings generated: {len(all_embeddings)}")
        # print(f"Dimension size per chunk: {len(all_embeddings[0])}")
        
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.embedding = embedding
        
        print("Embeddings successfully mapped back to text nodes!")
        
if __name__ == '__main__':
    main()
    