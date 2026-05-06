from typing import List, Dict
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import settings

VECTOR_SIZE = 3072


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL)


def ensure_collection_exists():
    """
    Qdrant 컬렉션이 없으면 생성합니다.
    Gemini embedding 모델은 3072차원 벡터를 반환합니다.
    """
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION_NAME

    collections = client.get_collections().collections
    collection_names = [
        collection.name
        for collection in collections
    ]

    if collection_name in collection_names:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )


def upsert_chunks(
        document_id: str,
        original_filename: str,
        chunks: List[Dict],
        embeddings: List[List[float]]
) -> int:
    """
    청크와 임베딩을 Qdrant에 저장합니다.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks와 embeddings 개수가 일치하지 않습니다.")

    ensure_collection_exists()

    client = get_qdrant_client()
    points = []

    for chunk, embedding in zip(chunks, embeddings):
        point = PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload={
                "document_id": document_id,
                "original_filename": original_filename,
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "page_chunk_index": chunk["page_chunk_index"],
                "text": chunk["text"],
                "text_length": chunk["text_length"]
            }
        )

        points.append(point)

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=points
    )

    return len(points)

def search_similar_chunks(
        question_embedding: List[float],
        limit: int = 5
) -> List[Dict]:
    """
    질문 임베딩과 유사한 청크를 Qdrant에서 검색합니다.
    """
    client = get_qdrant_client()

    query_result = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=question_embedding,
        limit=limit
    )

    results = []

    for point in query_result.points:
        payload = point.payload or {}

        results.append({
            "score": point.score,
            "document_id": payload.get("document_id"),
            "original_filename": payload.get("original_filename"),
            "chunk_id": payload.get("chunk_id"),
            "chunk_index": payload.get("chunk_index"),
            "page_number": payload.get("page_number"),
            "page_chunk_index": payload.get("page_chunk_index"),
            "text": payload.get("text"),
            "text_length": payload.get("text_length")
        })

    return results

def recreate_collection():
    """
    기존 Qdrant 컬렉션을 삭제하고 단일 벡터 컬렉션으로 다시 생성합니다.
    """
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION_NAME

    collections = client.get_collections().collections
    collection_names = [
        collection.name
        for collection in collections
    ]

    if collection_name in collection_names:
        client.delete_collection(
            collection_name=collection_name
        )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )