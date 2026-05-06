from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.chunker import create_chunks_from_pages
from app.services.embedding import create_embeddings
from app.services.pdf_loader import extract_pages_from_pdf

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    PDF 파일을 업로드하고 페이지별 텍스트를 추출합니다
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드할 수 있습니다."
        )

    stored_filename = f"{uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / stored_filename

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        pages = extract_pages_from_pdf(str(file_path))
        chunks = create_chunks_from_pages(
            pages=pages,
            chunk_size=800,
            overlap=120
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF 텍스트 추출 중 오류가 발생했습니다 : {str(e)}"
        )

    preview_pages = []

    for page in pages[:3]:
        preview_pages.append({
            "page_number": page["page_number"],
            "text_preview": page["text"][:500]
        })

    preview_chunks = []

    for chunk in chunks[:5]:
        preview_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "page_number": chunk["page_number"],
            "text_length": chunk["text_length"],
            "text_preview": chunk["text"][:300]
        })

    return {
        "original_filename" : file.filename,
        "stored_filename" : stored_filename,
        "page_count" : len(pages),
        "chunk_count" : len(chunks),
        "preview_pages" : preview_pages,
        "preview_chunks" : preview_chunks
    }

@router.post("/upload/embedding-test")
async def upload_document_embedding_test(file: UploadFile = File(...)):
    """
    PDF 파일을 업로드하고 청크 일부에 대해 임베딩을 생성합니다.
    Qdrant 저장 전 테스트용 API입니다.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드할 수 있습니다."
        )

    stored_filename = f"{uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / stored_filename

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        pages = extract_pages_from_pdf(str(file_path))
        chunks = create_chunks_from_pages(
            pages=pages,
            chunk_size=800,
            overlap=120
        )

        # 비용과 속도 관리를 위해 처음 3개 청크만 임베딩합니다.
        sample_chunks = chunks[:3]
        sample_texts = [
            chunk["text"]
            for chunk in sample_chunks
        ]

        embeddings = create_embeddings(sample_texts)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"임베딩 테스트 중 오류가 발생했습니다: {str(e)}"
        )

    preview_results = []

    for chunk, embedding in zip(sample_chunks, embeddings):
        preview_results.append({
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text_preview": chunk["text"][:200],
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:5]
        })

    return {
        "original_filename": file.filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "embedded_chunk_count": len(embeddings),
        "embedding_model": "gemini-embedding-001",
        "results": preview_results
    }