"""
ArthaSetu - RAG Ingestion Script

Reads product markdown files from data/products/, splits each one
section-wise (by ## headings), embeds each section using nomic-embed-text
via Ollama, and stores them in a persistent ChromaDB collection.

Run this once whenever you add or update a product file.
"""

import os
import re
import chromadb
from langchain_ollama import OllamaEmbeddings

PRODUCTS_DIR = "data/products"
CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "arthasetu_products"


def parse_sections(file_path: str, product_name: str) -> list[dict]:
    """
    Splits a product markdown file into chunks by ## heading.
    Returns a list of dicts: {text, product_name, section_type}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_sections = re.split(r"\n(?=## )", content)

    chunks = []
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue
        heading_match = re.match(r"## (.+)", section)
        if not heading_match:
            continue
        heading = heading_match.group(1).strip().lower()
        section_type = normalize_section_type(heading)

        chunks.append({
            "text": section,
            "product_name": product_name,
            "section_type": section_type,
        })
    return chunks


def normalize_section_type(heading: str) -> str:
    """Maps raw heading text to a consistent section_type tag."""
    heading = heading.lower()
    if "overview" in heading:
        return "overview"
    if "eligibility" in heading or "document" in heading:
        return "eligibility"
    if "interest" in heading or "return" in heading:
        return "rates"
    if "risk" in heading or "things to know" in heading:
        return "risks"
    if "how to apply" in heading or "apply" in heading:
        return "how_to_apply"
    return "other"


def load_all_products() -> list[dict]:
    """Walks data/products/ and parses every .md file found."""
    all_chunks = []
    for filename in os.listdir(PRODUCTS_DIR):
        if not filename.endswith(".md"):
            continue
        product_name = filename.replace(".md", "")
        file_path = os.path.join(PRODUCTS_DIR, filename)
        chunks = parse_sections(file_path, product_name)
        all_chunks.extend(chunks)
        print(f"  Parsed {filename}: {len(chunks)} sections")
    return all_chunks


def main():
    print("Loading product documents...")
    chunks = load_all_products()
    print(f"\nTotal chunks to embed: {len(chunks)}\n")

    print("Connecting to Ollama embeddings (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("Setting up ChromaDB persistent client...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    print("Embedding and storing chunks...")
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk["text"])
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[{
                "product_name": chunk["product_name"],
                "section_type": chunk["section_type"],
            }],
        )
        print(f"  [{i+1}/{len(chunks)}] {chunk['product_name']} - {chunk['section_type']}")

    print(f"\nDone. {len(chunks)} chunks stored in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()