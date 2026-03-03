# cli/lib/inverted_index.py

import os
import pickle

from .search_utils import load_movies
from .keyword_search import tokenize_text

class InvertedIndex:
    def __init__(self):
        self.index: dict[str,set[int]]={}
        self.docmap: dict[int,dict[list]]={}

    def __add_document(self,doc_id,text):
        tokenized_text=tokenize_text(text)
        for text in tokenized_text:
            if text not in self.index:
                self.index[text]=set()
            self.index[text].add(doc_id)
    
    def load(self) -> None:
        index_path = os.path.join(self.cache_dir, "index.pkl")
        docmap_path = os.path.join(self.cache_dir, "docmap.pkl")

        if not os.path.exists(index_path) or not os.path.exists(docmap_path):
            raise FileNotFoundError("Index files not found. Please run 'build' first.")

        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
            

    def get_documents(self, term):
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return list(doc_ids)
    
    def build(self) -> None:
        movies=load_movies()
        for movie in movies:
            docid=movie["id"]
            self.docmap[docid]=movie
            combined_text=f"{movie['title']} {movie['description']}"
            self.__add_document(docid,combined_text)

    def save(self)->None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "cache"
        )

        os.makedirs(cache_dir, exist_ok=True)

        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")

        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)


        
        
    