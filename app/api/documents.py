from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.chunker import create_chunks_from_pages
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