import re
from src.embed import get_embeddings
from src.vectordb import VectorDB


def normalize_query(query):
    # light normalization — lowercase, strip extra spaces
    query = query.strip().lower()
    query = re.sub(r'\s+', ' ', query)
    return query


def expand_query(query):
    query_lower = query.lower()
    expansions = {
        "under 18": "age restriction legally binding contract eligibility minor",
        "party to contracts": "role of ebay buyer seller contract transaction parties"
    }

    expanded_terms = [query]
    for key, expanded_text in expansions.items():
        if key in query_lower:
            expanded_terms.append(expanded_text)

    # combine into a single richer query string for embedding
    return " ".join(expanded_terms)


def retrieve_chunks(query, db, top_k=8):
    query = normalize_query(query)
    query = expand_query(query)
    query_vec = get_embeddings([query])

    if len(query_vec) == 0:
        return [], []

    results = db.search(query_vec, top_k=top_k)

    chunks = [r[0] for r in results]
    scores = [r[1] for r in results]

    return chunks, scores
