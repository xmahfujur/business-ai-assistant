import csv
import os

from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.readers.file import DocxReader, ImageReader, PyMuPDFReader

from src.config.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)
from src.rag.chunker import chunker


def _load_csv_as_documents(file_path: str) -> list[Document]:
    documents = []
    with open(file_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return documents

    headers = rows[0]
    for i, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        parts = []
        for header, value in zip(headers, row):
            if value.strip():
                parts.append(f"{header}: {value}")
        text = " | ".join(parts)
        documents.append(
            Document(
                text=text,
                metadata={
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    "row": i,
                },
            )
        )
    return documents


def universal_doc_loader(path: str) -> list:
    if not os.path.exists(path):
        return []

    file_extractors = {
        ".pdf": PyMuPDFReader(),
        ".docx": DocxReader(),
        ".png": ImageReader(keep_image=False),
        ".jpg": ImageReader(keep_image=False),
        ".jpeg": ImageReader(keep_image=False),
    }

    documents = []
    csv_files = []

    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        full_path = os.path.join(path, filename)
        if ext == ".csv":
            csv_files.append(full_path)
        else:
            loader = SimpleDirectoryReader(
                input_files=[full_path],
                file_extractor=file_extractors,
            )
            documents.extend(loader.load_data())

    for csv_path in csv_files:
        documents.extend(_load_csv_as_documents(csv_path))

    return documents


def get_collection_count() -> int:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = db.get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


def build_vector_index(rebuild: bool = False) -> VectorStoreIndex | None:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    from src.config.config import EMBED_MODEL

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    docs = universal_doc_loader(UPLOAD_DIR)
    if not docs:
        if rebuild:
            db = chromadb.PersistentClient(path=CHROMA_PATH)
            try:
                db.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
        return None

    chunks = chunker(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    if rebuild:
        try:
            db.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex(chunks, storage_context=storage_context)


if __name__ == "__main__":
    index = build_vector_index(rebuild=True)
    if index:
        print(f"Indexed {get_collection_count()} chunks into {CHROMA_PATH}")
