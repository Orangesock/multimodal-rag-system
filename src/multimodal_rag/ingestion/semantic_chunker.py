"""
Professional Structural Chunker with Multi-Vector Summary Logic.
Identifies document layout and generates retrieval-friendly summaries.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm

from multimodal_rag.resources import get_resources

logger = logging.getLogger(__name__)

def generate_chunk_summary(text: str) -> str:
    """
    Child-Vector Logic: Generates a dense semantic summary of a chunk.
    This helps the vector math 'find' lists and fragments that lack context.
    """
    if len(text.split()) < 10:
        return text

    resources = get_resources()
    generator = resources.get_generator_model()

    # We prompt the LLM to describe what this text IS, not just what it SAYS.
    prompt = f"Describe the following document section in one descriptive sentence for a search engine: {text[:500]}"
    
    try:
        output = generator(
            prompt,
            max_new_tokens=50,
            do_sample=False,
            num_beams=2
        )[0]["generated_text"]
        return output
    except Exception as e:
        logger.warning(f"Summary generation failed, using raw text: {e}")
        return text

def extract_structural_blocks(pdf_path: Path, output_image_dir: Path) -> List[Dict[str, Any]]:
    """
    Uses PyMuPDF 'blocks' with Geometric Column Sorting.
    Forces the parser to read the left column completely before the right column.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        raise RuntimeError(f"Cannot read PDF: {pdf_path}") from e

    pages_content = []
    pdf_image_dir = output_image_dir / pdf_path.stem
    pdf_image_dir.mkdir(parents=True, exist_ok=True)

    for page_num, page in enumerate(doc, start=1):
        # 1. Extract Images
        image_paths = []
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_filename = f"page_{page_num}_img_{img_index}.{base_image['ext']}"
            image_filepath = pdf_image_dir / image_filename
            with open(image_filepath, "wb") as f:
                f.write(base_image["image"])
            image_paths.append(str(image_filepath))

        # 2. Extract and Geometrically Sort Text Blocks
        blocks = page.get_text("blocks")
        page_width = page.rect.width

        left_column = []
        right_column = []
        full_width = []

        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0:  # Skip images/drawings in the text loop
                continue

            block_text = text.strip()
            if len(block_text) < 10 and not image_paths:
                continue

            block_width = x1 - x0
            block_center = x0 + (block_width / 2)

            # Geometric Sorting Logic
            # If the block spans more than 70% of the page, it's a Header/Footer
            if block_width > page_width * 0.7:
                full_width.append((y0, block_text))
            # If the center of the text is on the left half of the page
            elif block_center < page_width / 2:
                left_column.append((y0, block_text))
            # If the center is on the right half
            else:
                right_column.append((y0, block_text))

        # Sort each vertical column strictly from top to bottom (by y0 coordinate)
        left_column.sort(key=lambda x: x[0])
        right_column.sort(key=lambda x: x[0])
        full_width.sort(key=lambda x: x[0])

        # Merge in logical reading order: Headers -> Left Column -> Right Column
        ordered_texts = (
            [item[1] for item in full_width] +
            [item[1] for item in left_column] +
            [item[1] for item in right_column]
        )

        for block_text in ordered_texts:
            clean_text = re.sub(r"\s+", " ", block_text).strip()
            if clean_text:
                pages_content.append({
                    "page_num": page_num,
                    "text": clean_text,
                    "images": image_paths
                })

    doc.close()
    return pages_content

def load_and_semantic_chunk_pdfs(folder: str, progress=None) -> List[Dict[str, Any]]: # 🛠️ Added progress argument here
    """
    Main ingestion entry point. 
    Implements Parent-Child chunking by creating search summaries.
    """
    folder_path = Path(folder)
    image_output_path = folder_path / "images"

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = list(folder_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files. Starting structural ingestion...")

    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for pdf_file in sorted(pdf_files):
        blocks = extract_structural_blocks(pdf_file, image_output_path)
        total_blocks = len(blocks)
        
        logger.info(f"Generating search summaries for {pdf_file.name} (Parent-Child Logic)...")
        
        # We use a progress bar because generating summaries takes time
        for i, block in enumerate(tqdm(blocks, desc=f"Processing {pdf_file.name[:20]}")):
            
            # 🛠️ Update the Gradio UI progress bar
            if progress:
                progress((i + 1) / total_blocks, desc=f"AI is summarizing chunk {i+1} of {total_blocks}...")

            # Generate the 'Child' summary for retrieval math
            search_summary = generate_chunk_summary(block["text"])
            
            all_chunks.append({
                "source": str(pdf_file),
                "chunk_index": chunk_counter,
                "page": block["page_num"],
                "text": block["text"],           # The Parent (what the user reads)
                "search_text": search_summary,  # The Child (what the math embeds)
                "images": block["images"]
            })
            chunk_counter += 1

    logger.info(f"Created {len(all_chunks)} structural chunks with descriptive summaries.")
    return all_chunks