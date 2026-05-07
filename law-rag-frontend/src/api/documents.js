const BASE = import.meta.env.VITE_API_BASE_URL;

// PDF 업로드 + 인덱싱
export async function uploadAndIndex(file, maxChunks = 20) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/upload/index?max_chunks=${maxChunks}`, {
        method: "POST",
        body: form,
    });
    if (!res.ok) throw new Error("업로드 실패");
    return res.json();
}

// 질문 → RAG 답변
export async function askQuestion(question, limit = 5) {
    const res = await fetch(`${BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, limit }),
    });
    if (!res.ok) throw new Error("질문 요청 실패");
    return res.json();
}

// 유사 청크 검색 (답변 없이 검색만)
export async function searchDocuments(question, limit = 5) {
    const res = await fetch(`${BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, limit }),
    });
    if (!res.ok) throw new Error("검색 실패");
    return res.json();
}

// 저장된 벡터 수 확인
export async function getVectorCount() {
    const res = await fetch(`${BASE}/vector-store/count`);
    if (!res.ok) throw new Error("조회 실패");
    return res.json();
}
