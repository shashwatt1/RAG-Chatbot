from src.retriever import retrieve_chunks
from src.prompt import build_prompt
from src.generator import generate_response_stream
from src.vectordb import VectorDB


def answer_query(query, db, top_k=8, model="llama3.2"):
    chunks, scores = retrieve_chunks(query, db, top_k=top_k)

    # build prompt even if chunks is empty
    prompt = build_prompt(query, chunks)
    stream = generate_response_stream(prompt, model=model)

    return stream, chunks
