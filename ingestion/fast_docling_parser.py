"""
LIGHTWEIGHT Fast Parser for CA Study Material PDFs
(pdfplumber-only — no docling/PyTorch dependency)

Optimized for:
- Text-heavy CA PDFs (law, accounting, tax, audit)
- Accurate page number tracking per element
- Heading detection for section-level metadata
- Table extraction via pdfplumber

Drop-in replacement for the docling-based parser.
Processing time: 2-8 seconds per PDF.
"""

import re
import pdfplumber
from typing import List, Dict, Any
from pathlib import Path

print("⚡ FAST Parser loaded (pdfplumber-only, no docling)")


class FastDoclingParser:
    """
    Lightweight CA-optimized parser using pdfplumber only.

    Produces:
    - ✅ Text elements with accurate page numbers
    - ✅ Heading detection (numbered sections, ALLCAPS headings)
    - ✅ Table extraction
    - ❌ Images — intentionally skipped (CA PDFs are text/table only)
    """

    # ── Heading detection ─────────────────────────────────────────────────────

    @staticmethod
    def _is_heading(text: str, font_size: float = None, avg_font_size: float = None) -> bool:
        """
        Detect headings in CA legal/accounting documents.

        Matches:
          - Numbered sections: 1. Introduction / 2.3 Application
          - ALL-CAPS short lines: DEFINITIONS, PRELIMINARY
          - Clause-style: Section 2(1) / Clause (a)
          - Larger font size than average (when font info available)
        """
        t = text.strip()
        if not t or len(t) > 120:
            return False

        # Font size based detection (most reliable)
        if font_size and avg_font_size and font_size > avg_font_size * 1.1:
            return True

        # Numbered section headings: "1.", "2.3", "Section 1", "Clause 5"
        if re.match(r"^(\d+\.)+\s+\w", t):
            return True
        if re.match(r"^(section|clause|rule|schedule|chapter|part)\s+\d", t, re.IGNORECASE):
            return True

        # ALL CAPS heading (short, meaningful)
        words = t.split()
        if len(words) <= 8 and t == t.upper() and any(c.isalpha() for c in t) and len(t) > 3:
            return True

        return False

    # ── Table extraction ──────────────────────────────────────────────────────

    def extract_tables_fast(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables with page numbers via pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    for table_index, table in enumerate(page_tables or []):
                        if not table or len(table) < 2:
                            continue
                        cleaned = []
                        for row in table:
                            cleaned_row = [(cell.strip() if cell else "") for cell in row]
                            if any(cleaned_row):
                                cleaned.append(cleaned_row)
                        if len(cleaned) >= 2:
                            tables.append({
                                "page":        page_num,
                                "table_index": table_index,
                                "headers":     cleaned[0],
                                "rows":        cleaned[1:],
                                "num_rows":    len(cleaned) - 1,
                                "num_columns": len(cleaned[0]),
                                "raw_data":    cleaned,
                            })
        except Exception as e:
            print(f"  ⚠️  Table extraction error: {e}")
        return tables

    # ── Main parse ────────────────────────────────────────────────────────────

    def parse_pdf_fast(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF: extract text elements with page numbers + tables.
        Uses pdfplumber only — no ML models required.
        """
        print(f"  ⚡ Parsing: {Path(file_path).name}")

        elements = []
        total_pages = 0

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

                # First pass: collect all font sizes to compute average
                all_font_sizes = []
                for page in pdf.pages:
                    try:
                        chars = page.chars or []
                        all_font_sizes.extend([c.get("size", 0) for c in chars if c.get("size")])
                    except Exception:
                        pass

                avg_font_size = (sum(all_font_sizes) / len(all_font_sizes)) if all_font_sizes else 12.0

                # Second pass: extract text per page
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        # Extract words with font info grouped into lines
                        page_elements = self._extract_page_elements(page, page_num, avg_font_size)
                        elements.extend(page_elements)
                    except Exception as e:
                        print(f"  ⚠️  Page {page_num} extraction error: {e}")
                        # Fallback: plain text extract
                        text = page.extract_text() or ""
                        for para in text.split("\n\n"):
                            para = para.strip()
                            if para and len(para) > 20:
                                elements.append({
                                    "type": "paragraph",
                                    "text": para,
                                    "page": page_num,
                                })

        except Exception as e:
            print(f"  ⚠️  PDF open error: {e}")
            elements = [{"type": "paragraph", "text": "Could not extract text from document", "page": 1}]

        # Extract tables
        tables = self.extract_tables_fast(file_path)

        print(f"  ✅ Parsed: {len(elements)} elements, {len(tables)} tables, {total_pages} pages")

        return {
            "elements": elements,
            "tables":   tables,
            "images":   [],
            "metadata": {
                "total_elements": len(elements),
                "total_tables":   len(tables),
                "total_images":   0,
                "total_pages":    total_pages,
                "filename":       Path(file_path).name,
                "fast_mode":      True,
            },
        }

    def _extract_page_elements(
        self,
        page,
        page_num: int,
        avg_font_size: float,
    ) -> List[Dict[str, Any]]:
        """
        Extract structured text elements from a single page.
        Groups words into lines, then lines into paragraphs.
        Detects headings via font size + text patterns.
        """
        elements = []

        # Use extract_words for font-size aware grouping
        words = page.extract_words(
            extra_attrs=["size", "fontname"],
            keep_blank_chars=False,
        ) or []

        if not words:
            # Fallback to plain text
            text = page.extract_text() or ""
            for para in text.split("\n\n"):
                para = para.strip()
                if para and len(para) > 20:
                    elements.append({
                        "type": "paragraph",
                        "text": para,
                        "page": page_num,
                    })
            return elements

        # Group words into lines by their vertical position (top coordinate)
        lines: Dict[int, List] = {}
        for word in words:
            # Round top to nearest 3px to group words on the same line
            line_key = round(word.get("top", 0) / 3) * 3
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(word)

        # Sort lines top-to-bottom
        sorted_lines = sorted(lines.items(), key=lambda x: x[0])

        # Build line texts with average font size per line
        line_data = []
        for _, line_words in sorted_lines:
            line_words_sorted = sorted(line_words, key=lambda w: w.get("x0", 0))
            line_text = " ".join(w["text"] for w in line_words_sorted).strip()
            if not line_text:
                continue
            sizes = [w.get("size", avg_font_size) for w in line_words_sorted if w.get("size")]
            line_avg_size = sum(sizes) / len(sizes) if sizes else avg_font_size
            line_data.append((line_text, line_avg_size))

        # Group lines into paragraphs / headings
        current_para_lines = []
        current_para_size  = avg_font_size

        def flush_para():
            if not current_para_lines:
                return None
            text = " ".join(current_para_lines).strip()
            if len(text) < 15:
                return None
            is_head = self._is_heading(text, current_para_size, avg_font_size)
            return {
                "type": "heading" if is_head else "paragraph",
                "text": text,
                "page": page_num,
            }

        for line_text, line_size in line_data:
            is_head = self._is_heading(line_text, line_size, avg_font_size)

            if is_head:
                # Flush current paragraph first
                el = flush_para()
                if el:
                    elements.append(el)
                current_para_lines = []
                # Heading is its own element
                elements.append({
                    "type": "heading",
                    "text": line_text,
                    "page": page_num,
                })
                current_para_size = avg_font_size
            else:
                # Check if font size changed significantly (new paragraph)
                if current_para_lines and abs(line_size - current_para_size) > 1.5:
                    el = flush_para()
                    if el:
                        elements.append(el)
                    current_para_lines = []

                current_para_lines.append(line_text)
                current_para_size = line_size

        # Flush remaining paragraph
        el = flush_para()
        if el:
            elements.append(el)

        return elements


# ── Module-level convenience function ────────────────────────────────────────

def parse_pdf_fast(file_path: str) -> Dict[str, Any]:
    """Convenience wrapper for direct use."""
    parser = FastDoclingParser()
    return parser.parse_pdf_fast(file_path)
