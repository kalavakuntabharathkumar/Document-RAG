from pathlib import Path
from pypdf import PdfReader
import re

def extract_pdf_text(pdf_path: str | Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()

def chunk_text(text: str, chunk_words: int = 200, overlap: int = 40) -> list[str]:
    if chunk_words <= 0 or overlap < 0 or overlap >= chunk_words:
        raise ValueError("Require chunk_words > overlap >= 0.")
    words = re.findall(r"\S+", text)
    chunks = []
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks

def load_corpus(pdf_dir: str | Path, chunk_words: int = 200) -> list[dict]:
    records = []
    for path in sorted(Path(pdf_dir).glob("*.pdf")):
        text = extract_pdf_text(path)
        for index, chunk in enumerate(chunk_text(text, chunk_words)):
            records.append({
                "source": path.name,
                "chunk_id": index,
                "text": chunk,
            })
    return records
