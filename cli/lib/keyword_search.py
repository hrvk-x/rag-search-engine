import os
import pickle
import string
from collections import defaultdict
from collections import Counter
from nltk.stem import PorterStemmer
import math

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
    BM25_K1,
    BM25_B)
class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies:dict[int,Counter]={}
        self.doc_lengths={}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencey_path=os.path.join(CACHE_DIR,"tf.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencey_path,"wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths,f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencey_path,"rb") as f:
            self.term_frequencies=pickle.load(f)
        with open(self.doc_lengths_path,"rb") as f:
            self.doc_lengths=pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        doclength=len(tokens)
        self.doc_lengths[doc_id]=doclength

        if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id]=Counter()
            
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_tf(self,doc_id:int,term:str) -> int:
        tokens=tokenize_text(term)
        if len(tokens)!=1:
            raise ValueError(f"Expected single token , but received {len(tokens)} tokens")
        token=tokens[0]
        if doc_id in self.term_frequencies:
            return self.term_frequencies.get(doc_id, {}).get(token, 0)
        
    def get_bm25_idf(self, term: str) -> float:
        tokens=tokenize_text(term)
        if len(tokens)!=1:
            raise ValueError(f"Expected single token , but received {len(tokens)} tokens")
        token=tokens[0]
        N=len(self.docmap)
        df = len(self.index.get(token, set()))
        bm25=math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1,b = BM25_B) -> float:
        tf=self.get_tf(doc_id,term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        bm25tf=(tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25tf
        
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total_length = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        return total_length / num_docs

    def bm25(self, doc_id, term):
        score=self.get_bm25_tf(doc_id,term) * self.get_bm25_idf(term)
        return score

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        tokens = tokenize_text(query)
        scores: dict[int, float] = {}

        for token in tokens:
            # Get documents containing this token
            matching_docs = self.index.get(token, set())

            for doc_id in matching_docs:
                # Initialize score if not already present
                if doc_id not in scores:
                    scores[doc_id] = 0.0

                
                scores[doc_id] += self.bm25(doc_id, token)

        # Sort documents by score (highest first)
        ranked_docs = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # Return top `limit` results as (document, score)
        results = []
        for doc_id, score in ranked_docs[:limit]:
            results.append({
                "document": self.docmap[doc_id],
                "score": score
            })

        return results

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(id:int,text:str) -> int:
    idx=InvertedIndex()
    idx.load()
    result=idx.get_tf(id,text)
    return result

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def idf_command(text:str) -> float :
    idx=InvertedIndex()
    idx.load()
    tokens=tokenize_text(text)
    if len(tokens)!=1:
        raise ValueError(f"Expected single token , but received {len(tokens)} tokens")
    token=tokens[0]
    total_doc_count=len(idx.docmap)
    term_match_doc_count=len(idx.index.get(token,set()))
    idf=math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    return idf

def tfidf_command(docid:int,text:str) -> float:
    idx=InvertedIndex()
    idx.load()
    tokens=tokenize_text(text)
    if len(tokens)!=1:
        raise ValueError(f"Expected single token , but received {len(tokens)} tokens")
    token=tokens[0]
    total_doc_count=len(idx.docmap)
    term_match_doc_count=len(idx.index.get(token,set()))
    idf=math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    tf=idx.get_tf(docid,text)
    tfidf=tf*idf
    return tfidf

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def bm25_idf_command(term : str) -> float:
    idx=InvertedIndex()
    idx.load()
    score=idx.get_bm25_idf(term)
    return score

def bm25_tf_command(id:int,term : str,k1=BM25_K1,k2=BM25_B) -> float:
    idx=InvertedIndex()
    idx.load()
    score = idx.get_bm25_tf(id, term, k1, k2)
    return score

def bm25_search_command(query: str, limit: int = 5):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)