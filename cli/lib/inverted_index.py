import os
import pickle
from typing import Dict, Set, Any, List
from lib.search_utils import tokenize, load_movies, CACHE_PATH


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Any] = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize(text.lower())
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term) -> List[int]:
        term = term.lower()
        if term not in self.index:
            return []
        return list(sorted(self.index[term]))

    def build(self):
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_content = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, doc_content)
            self.docmap[doc_id] = doc_content

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        index_file = os.path.join(CACHE_PATH, "index.pkl")
        docmap_file = os.path.join(CACHE_PATH, "docmap.pkl")

        with open(index_file, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_file, "wb") as f:
            pickle.dump(self.docmap, f)
