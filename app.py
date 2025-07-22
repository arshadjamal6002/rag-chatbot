# app.py

import streamlit as st
from src.retriever import DocumentRetriever
from src.generator import stream_response

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄🔍 RAG Chatbot with Streaming Responses")

# Initialize
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = DocumentRetriever("vectordb")

# Sidebar Info
with st.sidebar:
    st.header("📊 System Info")
    st.markdown("**Model:** Mistral 7B (via LM Studio)")
    st.markdown(f"**# Chunks:** {len(st.session_state.retriever.chunks)}")
    if st.button("♻️ Clear Chat"):
        st.session_state.chat_history = []

# Input
user_query = st.text_input("Ask a question about the document:", key="user_input")

if user_query:
    with st.spinner("Retrieving relevant chunks..."):
        results = st.session_state.retriever.retrieve(user_query, top_k=3)
        prompt = st.session_state.retriever.format_prompt(user_query, results)

    # Display user message
    st.chat_message("user").markdown(user_query)

    # Streaming model output
    st.chat_message("assistant")
    with st.empty():
        full_response = ""
        for chunk in stream_response(prompt):
            full_response += chunk
            st.markdown(full_response + "▌")

        st.session_state.chat_history.append({"query": user_query, "response": full_response})

    # Show source chunks
    with st.expander("📎 Show Source Chunks"):
        for i, (chunk, score) in enumerate(results):
            st.markdown(f"**Chunk {i+1} (Score: {score:.2f})**")
            st.markdown(f"> {chunk}")

# Show chat history
if st.session_state.chat_history:
    st.subheader("📜 Chat History")
    for msg in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {msg['query']}")
        st.markdown(f"**Bot:** {msg['response']}")
        st.markdown("---")
