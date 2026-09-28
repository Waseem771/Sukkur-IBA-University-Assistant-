import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "IBA_Sukkur_University"
CHUNKS_DIR = "chunks"
INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 512


def load_pdfs(data_dir):
    root = Path(data_dir)
    pdf_files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdf_files.append(p)
    pdf_files.sort()
    print(f"[1/4] Found {len(pdf_files)} PDF files")

    if not pdf_files:
        print("Folder contents:")
        for p in list(root.rglob("*"))[:30]:
            print("  ", p)
        sys.exit("No PDFs found. Check DATA_DIR and the download step.")

    pages = []
    failed = []
    for pdf in tqdm(pdf_files, desc="Reading PDFs"):
        rel = pdf.relative_to(root)
        parts = rel.parts
        if len(parts) > 1:
            department = parts[0]
        else:
            department = "General"
        try:
            loaded = PyPDFLoader(str(pdf)).load()
        except Exception as e:
            failed.append(str(rel) + ": " + str(e)[:80])
            continue
        for p in loaded:
            text = p.page_content.strip()
            if not text:
                continue
            meta = {}
            meta["source"] = str(rel)
            meta["filename"] = pdf.name
            meta["department"] = department
            meta["folder_path"] = "/".join(parts[:-1])
            meta["page"] = int(p.metadata.get("page", 0)) + 1
            pages.append(Document(page_content=text, metadata=meta))

    print(f"      Pages with text: {len(pages)} | unreadable files: {len(failed)}")
    for f in failed:
        print("      -", f)
    return pages, failed


def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["char_count"] = len(c.page_content)
    print(f"[2/4] Created {len(chunks)} chunks")
    return chunks


def save_chunks(chunks):
    out = Path(CHUNKS_DIR)
    out.mkdir(exist_ok=True)

    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            row = {"text": c.page_content, "metadata": c.metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out / "chunks_preview.txt", "w", encoding="utf-8") as f:
        for c in chunks[:50]:
            m = c.metadata
            header = "=== chunk {} | {} | {} | p.{} ===\n".format(
                m["chunk_id"], m["department"], m["filename"], m["page"]
            )
            f.write(header)
            f.write(c.page_content + "\n\n")

    print(f"      Saved chunks to '{CHUNKS_DIR}/chunks.jsonl' and chunks_preview.txt")


def build_index(chunks):
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"[3/4] Creating embeddings on {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = None
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch = chunks[i:i + BATCH_SIZE]
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)

    db.save_local(INDEX_DIR)
    print(f"      Saved FAISS index to '{INDEX_DIR}/' (index.faiss + index.pkl)")


def save_metadata(chunks, pages, failed):
    out = Path(INDEX_DIR)

    all_meta = [c.metadata for c in chunks]
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    dept_chunks = Counter()
    dept_files = {}
    all_sources = set()
    for c in chunks:
        dept = c.metadata["department"]
        src = c.metadata["source"]
        dept_chunks[dept] += 1
        if dept not in dept_files:
            dept_files[dept] = set()
        dept_files[dept].add(src)
        all_sources.add(src)

    departments = {}
    for dept in sorted(dept_chunks):
        departments[dept] = {
            "chunks": dept_chunks[dept],
            "files": sorted(dept_files[dept]),
        }

    manifest = {}
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["embedding_model"] = EMBED_MODEL
    manifest["normalized_embeddings"] = True
    manifest["chunk_size"] = CHUNK_SIZE
    manifest["chunk_overlap"] = CHUNK_OVERLAP
    manifest["total_files"] = len(all_sources)
    manifest["total_pages"] = len(pages)
    manifest["total_chunks"] = len(chunks)
    manifest["departments"] = departments
    manifest["unreadable_files"] = failed
    manifest["filterable_fields"] = [
        "department", "source", "filename", "folder_path", "page"
    ]

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[4/4] Saved metadata.json and manifest.json")
    print("      Departments:", dict(dept_chunks))


def main():
    if not Path(DATA_DIR).exists():
        sys.exit(f"Folder '{DATA_DIR}' not found. Run the download cell first.")

    pages, failed = load_pdfs(DATA_DIR)
    if not pages:
        sys.exit("No text extracted. PDFs may be scanned images (need OCR).")

    chunks = create_chunks(pages)
    if not chunks:
        sys.exit("Chunking produced 0 chunks.")
    save_chunks(chunks)

    build_index(chunks)
    save_metadata(chunks, pages, failed)
    print("\nDone.")


if __name__ == "__main__":
    main()
