# # vector_index.py
# from typing import Dict, Any, List
# from sentence_transformers import SentenceTransformer

# # Load model once
# _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# def vector_index_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Step 9: Vector Index Construction (REAL embeddings - free HuggingFace)

#     Input:
#       - processed_chunks

#     Output:
#       - vector_index
#     """

#     processed = state.get("processed_chunks", [])
#     if not processed:
#         return {"vector_index": []}

#     texts = [c.get("text", "").strip() for c in processed]
#     texts = [t for t in texts if t]

#     if not texts:
#         return {"vector_index": []}

#     # Batch encode (faster)
#     vectors = _model.encode(texts, normalize_embeddings=True)

#     vector_index: List[Dict[str, Any]] = []
#     vec_i = 0

#     for chunk in processed:
#         text = chunk.get("text", "").strip()
#         if not text:
#             continue

#         embedding = vectors[vec_i].tolist()
#         vec_i += 1

#         vector_index.append({
#             "source": chunk.get("source"),
#             "url": chunk.get("url"),
#             "type": chunk.get("type", "unknown"),
#             "chunk_id": chunk.get("chunk_id"),
#             "text": text,
#             "embedding": embedding
#         })

#     return {"vector_index": vector_index}

# vector_index.py
from __future__ import annotations

from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer

# Load model once
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def vector_index_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 9: Vector Index Construction (REAL embeddings - HuggingFace)

    Input:
      - processed_chunks

    Output:
      - vector_index: List[Dict[str, Any]] with "embedding" as List[float]
    """
    processed = state.get("processed_chunks", [])
    if not processed:
        return {"vector_index": []}

    texts: List[str] = []
    for c in processed:
        t = (c.get("text") or "").strip()
        if t:
            texts.append(t)

    if not texts:
        return {"vector_index": []}

    # Batch encode
    vectors = _model.encode(texts, normalize_embeddings=True)

    vector_index: List[Dict[str, Any]] = []
    vec_i = 0

    for chunk in processed:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        emb = vectors[vec_i]
        vec_i += 1

        vector_index.append(
            {
                "source": chunk.get("source"),
                "url": chunk.get("url"),
                "type": chunk.get("type", "unknown"),
                "chunk_id": chunk.get("chunk_id"),
                "text": text,
                "embedding": emb.tolist(),
            }
        )

    return {"vector_index": vector_index}

