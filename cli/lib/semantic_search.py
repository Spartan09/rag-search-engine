from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]


def verify_model():
    semantic_search = SemanticSearch()
    model = semantic_search.model

    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
