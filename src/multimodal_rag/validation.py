"""Input validation for queries and PDF files."""

from pathlib import Path

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 1000
MAX_PDF_SIZE_MB = 50
MAX_TOTAL_CORPUS_SIZE_MB = 500


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def validate_query(query: str) -> str:
    """Validate and clean user query.

    Args:
        query: User's search query

    Returns:
        Cleaned query string

    Raises:
        ValidationError: If query is invalid
    """
    cleaned = query.strip()

    if not cleaned:
        raise ValidationError("Query cannot be empty")

    if len(cleaned) < MIN_QUERY_LENGTH:
        raise ValidationError(
            f"Query must be at least {MIN_QUERY_LENGTH} characters"
        )

    if len(cleaned) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Query cannot exceed {MAX_QUERY_LENGTH} characters"
        )

    return cleaned


def validate_pdf_file(file_path: Path) -> None:
    """Validate a single PDF file.

    Args:
        file_path: Path to PDF file

    Raises:
        ValidationError: If file is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValidationError(
            f"File must be a PDF: {file_path.name}"
        )

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_PDF_SIZE_MB:
        raise ValidationError(
            f"File {file_path.name} exceeds {MAX_PDF_SIZE_MB} MB limit"
        )


def validate_pdf_files(file_paths: list) -> None:
    """Validate multiple PDF files.

    Args:
        file_paths: List of Path objects to PDF files

    Raises:
        ValidationError: If any file is invalid
    """
    if not file_paths:
        raise ValidationError("No files provided")

    total_size_mb = 0
    for file_path in file_paths:
        validate_pdf_file(file_path)
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        total_size_mb += file_size_mb

    if total_size_mb > MAX_TOTAL_CORPUS_SIZE_MB:
        raise ValidationError(
            f"Total corpus size exceeds {MAX_TOTAL_CORPUS_SIZE_MB} MB limit"
        )


def get_config_limits() -> dict:
    """Get validation limits for UI display.

    Returns:
        Dictionary with configured limits
    """
    return {
        "min_query_length": MIN_QUERY_LENGTH,
        "max_query_length": MAX_QUERY_LENGTH,
        "max_pdf_size_mb": MAX_PDF_SIZE_MB,
        "max_total_corpus_size_mb": MAX_TOTAL_CORPUS_SIZE_MB,
    }
