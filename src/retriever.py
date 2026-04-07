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
        "party to contracts": "role of ebay buyer seller contract transaction parties",
        "responsibilities": "seller duties responsibilities listing delivery compliance policy obligations",
        "obligations": "seller duties responsibilities listing delivery compliance policy obligations",
        "duties": "seller duties responsibilities listing delivery compliance policy obligations"
    }

    expanded_terms = [query]
    for key, expanded_text in expansions.items():
        if key in query_lower:
            expanded_terms.append(expanded_text)

    # combine terms into one query string
    return " ".join(expanded_terms)


def retrieve_chunks(query, db, top_k=8):
    query = normalize_query(query)
    query = expand_query(query)
    query_vec = get_embeddings([query])

    if len(query_vec) == 0:
        return [], []

    # fetch extra chunks to ensure we get a diverse set of results
    raw_results = db.search(query_vec, top_k=25)

    final_results = []
    section_counts = {}

    for meta, score in raw_results:
        # group document chunks into general regions
        try:
            chunk_region = int(meta["chunk_id"]) // 4
        except ValueError:
            chunk_region = str(meta["chunk_id"])

        # limit to 2 chunks per region to avoid redundancy
        if section_counts.get(chunk_region, 0) < 2:
            final_results.append((meta, score))
            section_counts[chunk_region] = section_counts.get(chunk_region, 0) + 1

        if len(final_results) >= top_k:
            break

    chunks = [r[0] for r in final_results]
    scores = [r[1] for r in final_results]

    return chunks, scores
