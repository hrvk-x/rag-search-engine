#!/usr/bin/env python3

import argparse

from lib.keyword_search import build_command, search_command , tf_command , idf_command, tfidf_command , bm25_idf_command , bm25_tf_command,bm25_search_command
from lib.search_utils import BM25_K1,BM25_B

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser=subparsers.add_parser("tf",help="Find term frequency")
    tf_parser.add_argument("docid",type=int,help="Search the specific document")
    tf_parser.add_argument("term",type=str,help="Search specific term")

    idf_parser=subparsers.add_parser("idf",help="Inverse Document Frequency")
    idf_parser.add_argument("idf_term",type=str,help="Term for IDF")

    tfidf_parser=subparsers.add_parser("tfidf",help="TFIDF")
    tfidf_parser.add_argument("tfidf_docid",type=int,help="Search the specific document")
    tfidf_parser.add_argument("tfidf_term",type=str,help="Search specific term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser(
    "bm25search",
    help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    bm25search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            try:
                result=tf_command(args.docid,args.term)
                print(result)
            except ValueError as e:
                print(f"Input error :{e}")
        case "idf":
            try:
                value = idf_command(args.idf_term)
                print(f"{value:.2f}")
            except ValueError as e:
                print(f"Input error :{e}")
        
        case "tfidf":
            try:
                value = tfidf_command(args.tfidf_docid,args.tfidf_term)
                print(print(f"TF-IDF score of '{args.tfidf_term}' in document '{args.tfidf_docid}': {value:.2f}"))
            except ValueError as e:
                print(f"Input error :{e}")

        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except Exception as e:
                print(f"Input error :{e}")
        
        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id,args.term,args.k1)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except Exception as e:
                print(f"Input error :{e}")
        case "bm25search":
            print("BM25 Searching for:", args.query)
            try:
                results = bm25_search_command(args.query, args.limit)

                if not results:
                    print("No results found.")
                    return

                for i, result in enumerate(results, 1):
                    doc = result["document"]
                    score = result["score"]
                    print(f"{i}. ({doc['id']}) {doc['title']} | Score: {score:.2f}")

            except Exception as e:
                print(f"Search error: {e}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
