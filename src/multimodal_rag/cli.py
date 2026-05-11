"""CLI entrypoint - clean command interface."""

import typer
from multimodal_rag.config import configure_logging, settings
from multimodal_rag.embeddings.builder import build_index
from multimodal_rag.frontend.app import main as run_app

app = typer.Typer()


@app.command()
def build(
    index_type: str = typer.Option("flat", help="flat or ivf"),
):
    """Build FAISS index from PDFs in data/"""
    configure_logging()
    count = build_index(index_type=index_type)
    typer.echo(f"✅ Built index with {count} chunks")


@app.command()
def serve():
    """Launch Gradio web UI"""
    configure_logging()
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    run_app()


if __name__ == "__main__":
    app()