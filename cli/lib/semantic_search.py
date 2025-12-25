from .search_utils import DEFAULT_SEARCH_LIMIT
import os
from typing import Any
from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import CACHE_DIR, load_movies
import re


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.embeddings = None
        self.documents: list[Any] = None
        self.documents_map: dict[int, str] = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[Any]):
        self.documents = documents
        string_representations = []
        for doc in documents:
            self.documents_map[doc["id"]] = doc
            string_representations.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(
            string_representations, show_progress_bar=True
        )

        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.documents_map[doc["id"]] = doc

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")

    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def verify_model():
    semantic_search = SemanticSearch()
    model = semantic_search.model

    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    semantic_search = SemanticSearch()
    movies = load_movies()
    _ = semantic_search.load_or_create_embeddings(movies)
    return semantic_search.search(query, limit)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0):
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i : i + chunk_size]))

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunk_text(text: str, max_chunk_size: int = 200, overlap: int = 0):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for i in range(0, len(sentences), max_chunk_size - overlap):
        chunks.append(" ".join(sentences[i : i + max_chunk_size]))

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

    return chunks
