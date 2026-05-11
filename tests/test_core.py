"""Unit tests for multimodal RAG system core modules."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from multimodal_rag.validation import (
    validate_query,
    validate_pdf_file,
    ValidationError,
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    MAX_PDF_SIZE_MB,
)
from multimodal_rag.resources import ResourceManager, get_resources
from multimodal_rag.retrieval.bm25 import tokenize, build_bm25, search_bm25


class TestValidation:
    """Tests for input validation module."""

    def test_validate_query_empty(self):
        """Empty query should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_query("")

    def test_validate_query_whitespace(self):
        """Whitespace-only query should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_query("   ")

    def test_validate_query_too_short(self):
        """Query shorter than minimum should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_query("a")

    def test_validate_query_too_long(self):
        """Query longer than maximum should raise ValidationError."""
        long_query = "a" * (MAX_QUERY_LENGTH + 1)
        with pytest.raises(ValidationError):
            validate_query(long_query)

    def test_validate_query_valid(self):
        """Valid query should be returned cleaned."""
        result = validate_query("  What is this document about?  ")
        assert result == "What is this document about?"

    def test_validate_pdf_file_not_found(self):
        """Non-existent file should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_pdf_file(Path("/nonexistent/file.pdf"))

    def test_validate_pdf_file_wrong_extension(self, tmp_path):
        """Non-PDF file should raise ValidationError."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test")
        with pytest.raises(ValidationError):
            validate_pdf_file(txt_file)


class TestTokenization:
    """Tests for BM25 tokenization."""

    def test_tokenize_basic(self):
        """Basic tokenization should split on whitespace and lowercase."""
        tokens = tokenize("Hello World Test")
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_empty(self):
        """Empty string should return empty list."""
        tokens = tokenize("")
        assert tokens == []

    def test_tokenize_punctuation(self):
        """Punctuation should be included in tokens."""
        tokens = tokenize("Hello, world!")
        assert tokens == ["hello,", "world!"]


class TestResourceManager:
    """Tests for resource manager singleton pattern."""

    def test_singleton_pattern(self):
        """Multiple calls should return same instance."""
        rm1 = get_resources()
        rm2 = get_resources()
        assert rm1 is rm2

    def test_resource_manager_initialization(self):
        """Resource manager should initialize without error."""
        rm = ResourceManager()
        assert rm is not None
        assert rm._embedding_model is None  # Not loaded yet
        assert rm._reranker_model is None

    @patch("multimodal_rag.resources.SentenceTransformer")
    def test_get_embedding_model_caching(self, mock_sbert):
        """Embedding model should be cached after first call."""
        rm = ResourceManager()
        rm._embedding_model = None  # Reset

        model1 = rm.get_embedding_model()
        model2 = rm.get_embedding_model()

        # Should only be called once due to caching
        assert mock_sbert.call_count == 1
        assert model1 is model2


class TestBM25Search:
    """Tests for BM25 search functionality."""

    def test_build_bm25_empty_corpus(self):
        """Building BM25 with empty corpus should raise ValueError."""
        with pytest.raises(ValueError):
            build_bm25([])

    def test_build_bm25_valid(self):
        """Building BM25 with valid corpus should succeed."""
        corpus = ["Hello world", "Goodbye world", "Hello there"]
        bm25, tokenized = build_bm25(corpus)
        assert bm25 is not None
        assert len(tokenized) == 3

    def test_search_bm25_empty_query(self):
        """Searching with empty query should return empty results."""
        corpus = ["Hello world"]
        bm25, _ = build_bm25(corpus)
        metadata = [{"text": "Hello world", "source": "test.pdf", "chunk_index": 0}]

        results = search_bm25("", bm25, corpus, metadata)
        assert results == []

    def test_search_bm25_valid(self):
        """Valid BM25 search should return results."""
        corpus = ["Hello world", "Goodbye world"]
        bm25, _ = build_bm25(corpus)
        metadata = [
            {"text": "Hello world", "source": "test.pdf", "chunk_index": 0},
            {"text": "Goodbye world", "source": "test.pdf", "chunk_index": 1},
        ]

        results = search_bm25("hello", bm25, corpus, metadata, top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_index"] == 0


class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline (requires pre-built indexes)."""

    @pytest.mark.integration
    def test_rag_query_with_documents(self):
        """RAG query should work with loaded documents."""
        # This test assumes embeddings have been built
        try:
            from multimodal_rag.generation.rag_pipeline import rag_query

            # Should not raise exception even if no answer found
            result = rag_query("What is machine learning?")
            
            # 🔥 FIX: Assert it returns a dictionary with 'answer' and 'images'
            assert isinstance(result, dict)
            assert "answer" in result
            assert "images" in result
            assert isinstance(result["answer"], str)
            
        except FileNotFoundError:
            pytest.skip("Embeddings not built")

    @pytest.mark.integration
    def test_rag_query_invalid_query(self):
        """RAG query with invalid query should return error message."""
        try:
            from multimodal_rag.generation.rag_pipeline import rag_query

            result = rag_query("")
            
            # 🔥 FIX: Check the 'answer' key in the result dictionary
            assert isinstance(result, dict)
            answer_text = result.get("answer", "")
            assert "Invalid" in answer_text or "don't know" in answer_text
            
        except FileNotFoundError:
            pytest.skip("Embeddings not built")


@pytest.fixture
def temp_pdf_file(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test.pdf"
    # Create a dummy PDF (won't be valid but has .pdf extension)
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
    return pdf_path


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return [
        {
            "source": "test1.pdf",
            "chunk_index": 0,
            "page": 1,
            "text": "This is a test document about machine learning."
        },
        {
            "source": "test1.pdf",
            "chunk_index": 1,
            "page": 2,
            "text": "Machine learning is a type of artificial intelligence."
        },
        {
            "source": "test2.pdf",
            "chunk_index": 2,
            "page": 1,
            "text": "Deep learning uses neural networks for learning."
        },
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])