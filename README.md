# Modular RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions strictly based on your uploaded PDFs. It runs a local LLM via Ollama connected to a local vector store, meaning your data stays completely offline and private.

## Features
- Ask questions directly against uploaded documents
- Semantic search powered by local embeddings
- Strict grounding (zero hallucination allowed by the prompt)
- Hard fallbacks if it can't find the answer
- Clean, minimal Streamlit UI
- Collapsible source tracking
- Formatted output with auto-generated headers and bullet points
- Real-time token streaming

## How It Works
The pipeline keeps the LLM grounded:

**User Query → Embedding → Vector Search → Fetch Chunks → LLM → Stream Response**

- **Chunking:** We parse the document and chunk it by sentence bounds (~200 words) with overlapping text so we don't lose context between paragraphs.
- **Embeddings:** Text chunks get converted into dense vectors.
- **Vector DB:** We dump these vectors into FAISS for super fast cosine similarity lookups.
- **LLM Engine:** We pass the highest scoring chunks into a local Ollama model (like Mistral) via a highly restrictive prompt to generate the final answer.

## Tech Stack
- **Python**: Backend logic
- **Streamlit**: Fast frontend UI
- **FAISS**: Local vector engine
- **Sentence Transformers**: Generates embeddings locally
- **Ollama**: Local, offline LLM inference
- **NumPy**: Vector math

## Project Structure
```text
/data               # Drop raw docs here
/src                # Pipeline code (retriever, pipeline, prompt, database)
app.py              # Streamlit entry point
requirements.txt    # Dependencies
```

## Running Locally

1. Clone this repo.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Start up Ollama and grab a model (using `mistral` as an example):
   ```bash
   ollama pull mistral
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. Hit the `localhost` URL in your browser.

## Sample Queries
Upload a document and try testing the constraints:
- What is eBay?
- What are the responsibilities of sellers?
- What happens if a user violates policies?
- Does eBay provide insurance?

## Notes on Output
- **Grounded Answers**: The model is forced to only use the uploaded file.
- **Source Verification**: Open the "📄 Show Sources" expander to see the exact text chunks it used.
- **Fallbacks**: If it can't find what you asked for, it will instantly tell you instead of guessing.

## Current Limitations
- **Retrieval Bottlenecks**: If a concept spans dozens of non-adjacent pages, the chunk limit might miss some nuance.
- **Small Models**: Running this via heavily quantized local models can occasionally result in slightly varied formatting across multiple runs.

## Future Plans
- Swap in heavier embedding architectures for deeper semantic matching.
- Add hybrid search (keywords + semantics).
- Expand support for querying multiple PDFs at once.

---
**Shashwat Malviya**  
*AI/ML Engineer*
