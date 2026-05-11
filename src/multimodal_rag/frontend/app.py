"""Gradio frontend for the Multimodal RAG system."""

from __future__ import annotations

import logging
import shutil
from numbers import Number
from pathlib import Path
from typing import Iterable, List, Optional

import gradio as gr

from multimodal_rag.config import configure_logging, settings
from multimodal_rag.embeddings.builder import build_index
from multimodal_rag.generation.rag_pipeline import rag_query
from multimodal_rag.resources import get_resources
from multimodal_rag.validation import (
    ValidationError,
    get_config_limits,
    validate_pdf_files,
)

logger = logging.getLogger(__name__)


# =========================
# FILE HELPERS
# =========================

def _ensure_data_dir() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)


def _unique_destination(folder: Path, filename: str) -> Path:
    """
    Overwrites files with the same name. 
    This prevents the system from re-indexing duplicates of the same PDF.
    """
    candidate = folder / filename
    
    # If the file already exists, we delete it before copying the new one
    if candidate.exists():
        logger.info(f"Overwriting existing file to prevent duplicates: {candidate.name}")
        candidate.unlink() 
        
    return candidate


def _copy_uploaded_pdfs(uploaded_files: Optional[Iterable[str]]) -> List[str]:
    """
    Copy uploaded PDFs into the configured data directory.

    Gradio `File(type="filepath", file_count="multiple")` passes a list of paths.
    """
    if not uploaded_files:
        return []

    _ensure_data_dir()

    src_paths = [Path(p) for p in uploaded_files]
    validate_pdf_files(src_paths)

    copied_names: List[str] = []

    for src in src_paths:
        dst = _unique_destination(settings.data_dir, src.name)
        shutil.copy2(src, dst)
        copied_names.append(dst.name)

    return copied_names


def _clear_response_cache() -> None:
    """Remove cached answers so reindexing doesn't serve stale responses."""
    cache_dir = settings.base_dir / "cache" / "responses"
    if cache_dir.exists():
        for item in cache_dir.glob("*.pkl"):
            try:
                item.unlink()
            except Exception as exc:
                logger.warning("Could not delete cache file %s: %s", item, exc)


# =========================
# DISPLAY HELPERS
# =========================

def _fmt_score(value) -> str:
    if isinstance(value, Number):
        return f"{float(value):.3f}"
    return "n/a"


def format_sources(documents: list, max_items: int = 5) -> str:
    """Render retrieved documents as markdown."""
    if not documents:
        return "### Sources\nNo retrieved sources."

    lines = ["### Sources"]

    for i, doc in enumerate(documents[:max_items], start=1):
        source = Path(str(doc.get("source", "unknown"))).name
        page = doc.get("page", "n/a")
        score = doc.get("rerank_score", doc.get("score"))
        text = " ".join(str(doc.get("text", "")).split())
        preview = text[:350] + ("..." if len(text) > 350 else "")

        lines.append(
            f"**{i}. {source}** \n"
            f"Page: `{page}` · Score: `{_fmt_score(score)}`  \n"
            f"{preview}"
        )

    return "\n\n".join(lines)


def format_trace(result: dict) -> str:
    """Render pipeline trace as markdown."""
    trace = result.get("trace")
    cached = bool(result.get("cached", False))

    lines = ["### Trace"]

    if cached:
        lines.append("Answer served from cache.")
        return "\n\n".join(lines)

    if not trace:
        lines.append("No trace available.")
        return "\n\n".join(lines)

    total_time = trace.get("total_time")
    if isinstance(total_time, Number):
        lines.append(f"Total time: `{float(total_time):.2f}s`")

    steps = trace.get("steps", [])
    if not steps:
        lines.append("No recorded steps.")
        return "\n\n".join(lines)

    step_lines = []
    for step in steps:
        name = step.get("step", "unknown")
        elapsed = step.get("elapsed")
        data = step.get("data", {})
        if isinstance(elapsed, Number):
            elapsed_text = f"{float(elapsed):.2f}s"
        else:
            elapsed_text = "n/a"
        step_lines.append(f"- **{name}** at `{elapsed_text}` · `{data}`")

    lines.append("\n".join(step_lines))
    return "\n\n".join(lines)


def limits_markdown() -> str:
    limits = get_config_limits()
    return (
        "### Limits\n"
        f"- Min query length: `{limits['min_query_length']}`\n"
        f"- Max query length: `{limits['max_query_length']}`\n"
        f"- Max PDF size: `{limits['max_pdf_size_mb']} MB`\n"
        f"- Max total corpus size: `{limits['max_total_corpus_size_mb']} MB`"
    )


# =========================
# EVENT HANDLERS
# =========================

def handle_message(message: str, history: Optional[list]) -> tuple:
    """Submit a user query and return chat history + diagnostics."""
    history = history or []
    message = (message or "").strip()

    if not message:
        return "", history, "### Sources\nNo query submitted.", "### Trace\nNo query submitted."

    result = rag_query(message)
    answer = result["answer"]

    # --- GRADIO 6 DICTIONARY FORMAT ---
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    # ----------------------------------

    sources_md = format_sources(result.get("reranked") or result.get("retrieved") or [])
    trace_md = format_trace(result)

    return "", history, sources_md, trace_md


def handle_clear() -> tuple:
    """Clear the chat and side panels."""
    return [], "### Sources\nCleared.", "### Trace\nCleared."


def handle_upload_and_index(uploaded_files: Optional[Iterable[str]]) -> str:
    """Copy uploads into data/ and rebuild the FAISS/BM25 indexes."""
    try:
        copied = _copy_uploaded_pdfs(uploaded_files)

        if copied:
            copied_text = ", ".join(copied)
        else:
            copied_text = "No new files uploaded."

        _ensure_data_dir()
        
        count = build_index(data_dir=settings.data_dir, index_type="flat")

        # Force the singleton to reload the refreshed indexes and clear stale answers.
        get_resources().reload_indexes()
        _clear_response_cache()

        return (
            f"✅ {copied_text}\n\n"
            f"✅ Indexed `{count}` chunks successfully from `{settings.data_dir}`"
        )

    except ValidationError as exc:
        return f"❌ Validation error: {exc}"
    except Exception as exc:
        logger.exception("Index build failed")
        return f"❌ Index build failed: {exc}"


# =========================
# APP
# =========================

def main():
    """Launch the Gradio UI."""
    configure_logging()
    _ensure_data_dir()

    with gr.Blocks(
        title="Multimodal RAG",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 📄 Multimodal RAG")
        gr.Markdown(
            "Ask questions about your PDFs, upload new documents, and rebuild the index from the UI."
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Assistant",
                    height=560,
                )
                query = gr.Textbox(
                    label="Ask a question",
                    placeholder="What does the document say about ... ?",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                upload = gr.File(
                    label="Upload PDFs",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                index_btn = gr.Button("Upload and Reindex", variant="primary")
                upload_status = gr.Markdown("Upload PDFs, then rebuild the index.")
                sources_box = gr.Markdown("### Sources\nNo query yet.")
                trace_box = gr.Markdown("### Trace\nNo query yet.")
                gr.Markdown(limits_markdown())

        # Submit handlers
        send_btn.click(
            handle_message,
            inputs=[query, chatbot],
            outputs=[query, chatbot, sources_box, trace_box],
        )
        query.submit(
            handle_message,
            inputs=[query, chatbot],
            outputs=[query, chatbot, sources_box, trace_box],
        )

        # Clear chat
        clear_btn.click(
            handle_clear,
            inputs=[],
            outputs=[chatbot, sources_box, trace_box],
        ).then(
            lambda: "",
            inputs=[],
            outputs=[query],
        )

        # Upload / index
        index_btn.click(
            handle_upload_and_index,
            inputs=[upload],
            outputs=[upload_status],
        )

        demo.queue(default_concurrency_limit=2)

    demo.launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        share=settings.app_share,
    )


if __name__ == "__main__":
    main()