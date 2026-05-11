"""
Precision RAG Pipeline: Direct Answer Logic.
"""

import logging
from typing import Dict, Any, List

from multimodal_rag.validation import validate_query, ValidationError
from multimodal_rag.retrieval.hybrid import hybrid_search
from multimodal_rag.reranking.cross_encoder import rerank
from multimodal_rag.config import settings
from multimodal_rag.resources import get_resources
from multimodal_rag.faithfulness import enforce_faithfulness
from multimodal_rag.observability import Trace, log_query, log_answer

logger = logging.getLogger(__name__)

def clean_answer(text: str) -> str:
    return text.strip()

def reconstruct_page_context(reranked_docs: list, all_metadata: list) -> list:
    page_groups = {}
    for doc in reranked_docs:
        key = (doc["source"], doc["page"])
        if key not in page_groups:
            page_groups[key] = []
        page_groups[key].append(doc["chunk_index"])

    reconstructed_docs = []
    meta_map = {m["chunk_index"]: m for m in all_metadata}

    for (source, page_num), indices in page_groups.items():
        min_idx, max_idx = min(indices), max(indices)
        start, end = max(0, min_idx - 1), max_idx + 2
        
        full_text_parts = []
        images = []
        for i in range(start, end):
            if i in meta_map and meta_map[i]["source"] == source:
                full_text_parts.append(meta_map[i]["text"])
                if "images" in meta_map[i]:
                    images.extend(meta_map[i]["images"])

        reconstructed_docs.append({
            "source": source,
            "page": page_num,
            "text": "\n".join(full_text_parts),
            "images": list(set(images))
        })
    return reconstructed_docs

def build_prompt(query: str, documents: list) -> str:
    """Final Precision Prompt."""
    context_blocks = []
    for doc in documents:
        source_name = doc["source"].split("/")[-1]
        context_blocks.append(f"--- DOCUMENT SECTION (Source: {source_name}, Page: {doc['page']}) ---\n{doc['text']}")

    context = "\n\n".join(context_blocks)

    return f"""
Instructions:
1. Provide a direct and accurate answer based ONLY on the context below.
2. If the answer is short, keep it short. Do not add background info or 'filler' text.
3. If the answer requires a long explanation, provide detail.
4. Stop immediately after answering the question.

CONTEXT:
{context}

QUESTION:
{query}
"""

def generate_answer(prompt: str) -> str:
    resources = get_resources()
    generator = resources.get_generator_model()
    # 🛠️ MAX TOKENS is a ceiling, not a requirement. The AI will stop naturally.
    output = generator(prompt, max_new_tokens=512)
    return output[0]["generated_text"]

def rag_query(query: str) -> Dict[str, Any]:
    resources = get_resources()
    # Cache Check (Always good to have)
    cached = resources.response_cache.get(query)
    if cached: return cached

    trace = Trace()

    try:
        query = validate_query(query)
    except ValidationError:
        return {"answer": "Invalid query.", "images": []}

    log_query(query)
    trace.log_step("validated_query")

    _, metadata, _, _ = resources.get_retrieval_resources()
    retrieved = hybrid_search(query, top_k=15)
    if not retrieved:
        return {"answer": "No relevant info found.", "images": []}
    
    reranked = rerank(query, retrieved, top_k=5)
    trace.log_step("retrieval_and_rerank")

    merged_docs = reconstruct_page_context(reranked, metadata)
    trace.log_step("context_reconstructed")

    images = list(set([img for doc in merged_docs for img in doc.get("images", [])]))

    prompt = build_prompt(query, merged_docs)
    answer = generate_answer(prompt)
    answer = clean_answer(answer)
    trace.log_step("generation")

    # This will now correctly validate short answers
    answer = enforce_faithfulness(answer, merged_docs)
    log_answer(answer)

    result = {
        "answer": answer,
        "images": images,
        "reranked": reranked,
        "trace": trace.summary()
    }
    resources.response_cache.set(query, result)
    return result