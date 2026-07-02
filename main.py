import os
import pathlib
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config.config import SUPPORTED_EXTENSIONS, UPLOAD_DIR
from src.rag.engine import get_collection_count, query_documents, rebuild_index

app = FastAPI(
    title="Business AI Assistant API",
    description="Upload documents and chat with your knowledge base.",
    version="1.0.0",
)

UPLOAD_PATH = pathlib.Path(UPLOAD_DIR)
UPLOAD_PATH.mkdir(parents=True, exist_ok=True)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": get_collection_count(),
        "upload_dir": str(UPLOAD_PATH),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if get_collection_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload files via POST /uploads first.",
        )
    try:
        result = query_documents(request.question)
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.post("/uploads")
async def upload_documents(file: UploadFile = File(...)):
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    file_path = UPLOAD_PATH / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to save uploaded file.")

    try:
        count = rebuild_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File saved but indexing failed: {e}")

    return {
        "message": "Document uploaded and indexed successfully.",
        "filename": file.filename,
        "saved_to": str(file_path),
        "indexed_chunks": count,
    }


@app.post("/rebuild-index")
def rebuild():
    try:
        count = rebuild_index()
        return {"message": "Index rebuilt.", "indexed_chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
