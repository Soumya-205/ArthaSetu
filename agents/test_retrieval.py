"""
ArthaSetu - Quick retrieval test

Lets you type a question and see which chunks ChromaDB retrieves,
so you can sanity-check retrieval quality before building more on top.
"""

import chromadb
from langchain_ollama import OllamaEmbeddings

CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "arthasetu_products"


def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print("Type a question (or 'quit' to exit)\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit"):
            break

        query_vector = embeddings.embed_query(query)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3,
        )

        print("\nTop matches:")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            print(f"\n  #{i+1} | product: {meta['product_name']} | section: {meta['section_type']} | distance: {dist:.3f}")
            print(f"  {doc[:150]}...")
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()