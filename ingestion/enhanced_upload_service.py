"""
Enhanced Upload Service for CA Study Materials (IMAGE-FREE)

Complete ingestion pipeline with:
- Advanced PDF parsing (Docling + pdfplumber)
- Table extraction and structuring
- Optimized chunking for longer CA content
- Text + table embedding only (no images)
"""

from typing import Dict, Any, List, Optional
import uuid
import os
from pathlib import Path

# Import enhanced processors
from .enhanced_chunking import create_chunks_enhanced
from .enhanced_table_processor import process_table_for_embedding
from .metadata_builder import build_metadata
from .embedding_service import embed_texts
from .pinecone_service import upsert_vectors
from .fast_docling_parser import FastDoclingParser


async def process_pdf_enhanced(
    file_path: str, 
    file_name: str, 
    extra_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhanced PDF processing pipeline (no image processing)
    
    Args:
        file_path: Path to PDF file
        file_name: Original filename
        extra_meta: Metadata from admin (course, subject, chapter, etc.)
    
    Returns:
        dict with processing statistics and results
    """
    
    print(f"📄 Processing PDF: {file_name}")
    
    # ============================================================
    # STEP 1: Parse PDF with Fast Parser
    # ============================================================
    print("  ├─ Parsing PDF structure...")
    parser = FastDoclingParser()

    parse_result = parser.parse_pdf_fast(file_path)
    
    elements = parse_result["elements"]
    tables = parse_result["tables"]
    
    print(f"  ├─ Found {len(elements)} elements, {len(tables)} tables")
    
    # ============================================================
    # STEP 2: Create Enhanced Chunks
    # ============================================================
    print("  ├─ Creating optimized chunks...")
    
    chunks = create_chunks_enhanced(
        elements=elements,
        images=None,
        tables=tables
    )
    
    print(f"  ├─ Created {len(chunks)} text chunks")
    
    # ============================================================
    # STEP 3: Process Tables into Separate Chunks
    # ============================================================
    table_chunks = []
    
    if tables:
        print("  ├─ Processing tables...")
        for table in tables:
            table_text = process_table_for_embedding(table)
            
            if table_text and len(table_text) > 100:  # Only meaningful tables
                table_chunks.append({
                    "text": table_text,
                    "heading": f"Table on Page {table.get('page', 'unknown')}",
                    "page_start": table.get("page"),
                    "page_end": table.get("page"),
                    "length": len(table_text),
                    "type": "table",
                    "table_metadata": {
                        "num_rows": table.get("num_rows", 0),
                        "num_columns": table.get("num_columns", 0),
                        "headers": table.get("headers", [])
                    }
                })
        
        print(f"  ├─ Processed {len(table_chunks)} table chunks")
    
    # ============================================================
    # STEP 4: Combine All Chunks (text + tables only)
    # ============================================================
    all_chunks = chunks + table_chunks
    
    print(f"  ├─ Total chunks for embedding: {len(all_chunks)}")
    
    # ============================================================
    # STEP 5: Generate Embeddings
    # ============================================================
    print("  ├─ Generating embeddings...")
    
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = await embed_texts(texts)
    
    print(f"  ├─ Generated {len(embeddings)} embeddings")
    
    # ============================================================
    # STEP 6: Build Vectors with Metadata
    # ============================================================
    print("  ├─ Building vectors...")
    
    vectors = []
    
    for chunk, embedding in zip(all_chunks, embeddings):
        # Build metadata
        metadata = build_metadata(chunk, file_name, extra_meta)
        
        # Add chunk type
        metadata["chunk_type"] = chunk.get("type", "text")
        
        # Add table-specific metadata if present
        if "table_metadata" in chunk:
            metadata["table_rows"] = chunk["table_metadata"].get("num_rows")
            metadata["table_columns"] = chunk["table_metadata"].get("num_columns")
        
        # Create vector
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": metadata
        })
    
    # ============================================================
    # STEP 7: Upsert to Pinecone
    # ============================================================
    print("  ├─ Uploading to Pinecone...")
    
    upsert_vectors(vectors)
    
    print(f"  └─ ✅ Completed! Indexed {len(vectors)} vectors")
    
    # ============================================================
    # Return Statistics
    # ============================================================
    return {
        "total_vectors": len(vectors),
        "text_chunks": len(chunks),
        "table_chunks": len(table_chunks),
        "filename": file_name,
        "metadata": extra_meta
    }


async def process_pdf(
    file_path: str, 
    file_name: str, 
    extra_meta: Dict[str, Any]
) -> int:
    """
    Backward compatible wrapper for existing code
    
    Returns:
        Number of vectors created
    """
    result = await process_pdf_enhanced(file_path, file_name, extra_meta)
    return result["total_vectors"]


async def process_pdf_batch(
    file_paths: List[str],
    file_names: List[str],
    extra_metas: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process multiple PDFs in batch
    
    Useful for bulk uploads
    """
    results = []
    
    for file_path, file_name, extra_meta in zip(file_paths, file_names, extra_metas):
        try:
            result = await process_pdf_enhanced(file_path, file_name, extra_meta)
            results.append(result)
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            results.append({
                "filename": file_name,
                "error": str(e),
                "total_vectors": 0
            })
    
    return results
