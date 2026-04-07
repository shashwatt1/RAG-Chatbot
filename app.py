import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.preprocess import process_document
from src.embed import get_embeddings, get_dimension
from src.vectordb import VectorDB
from src.pipeline import answer_query

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# initialize database once per session
if "db" not in st.session_state:
    st.session_state.db = VectorDB(dimension=get_dimension())
    st.session_state.db.load()

if "messages" not in st.session_state:
    st.session_state.messages = []


# sidebar layout
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Ollama Model",
        ["llama3.2", "mistral", "phi3", "llama3"],
        index=0,
        help="Make sure you've run `ollama pull <model>` first"
    )
    
    st.divider()
    st.header("Database Info")
    chunks_count = st.session_state.db.index.ntotal if st.session_state.db.index else 0
    st.caption(f"**Active Model:** {model}")
    st.caption(f"**Index:** {chunks_count} chunks")
    st.caption("**Vector Engine:** FAISS CPU")

    st.divider()
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                st.info("Chunking text...")
                chunks = process_document(tmp_path)

                st.info(f"Embedding {len(chunks)} chunks...")
                texts = [c["text"] for c in chunks]
                embeddings = get_embeddings(texts)

                st.info("Indexing...")
                # create fresh index for new documents
                st.session_state.db = VectorDB(dimension=get_dimension())
                st.session_state.db.build_or_update(embeddings, chunks)

                st.success(f"Ready! {len(chunks)} chunks indexed.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
            finally:
                os.remove(tmp_path)

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# main chat interface
st.title("Modular RAG Chatbot")
st.markdown("Answers are generated dynamically based strictly on the uploaded document context.")

# show chat history and sources
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Show Sources"):
                for i, s in enumerate(msg["sources"][:3]):
                    preview = s.get("text", "")[:300].replace("\n", " ").strip() + "..."
                    st.markdown(f"**Source {i+1}:**\n{preview}\n")


if prompt := st.chat_input("Ask something about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        db = st.session_state.db

        if db.index is None or db.index.ntotal == 0:
            st.warning("No document indexed yet — upload a PDF from the sidebar.")
            full_response = "No document loaded."
            sources = []
            st.markdown(full_response)
        else:
            # stream the response to the ui
            stream, sources = answer_query(prompt, db, top_k=8, model=model)
            full_response = st.write_stream(stream)
            
            if sources:
                with st.expander("📄 Show Sources"):
                    for i, s in enumerate(sources[:3]):
                        preview = s.get("text", "")[:300].replace("\n", " ").strip() + "..."
                        st.markdown(f"**Source {i+1}:**\n{preview}\n")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
    })
