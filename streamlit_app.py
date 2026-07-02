import os
import pathlib
import shutil

import streamlit as st

from src.config.config import GEMINI_API_KEY, SUPPORTED_EXTENSIONS, UPLOAD_DIR
from src.rag.engine import get_collection_count, query_documents, rebuild_index

st.set_page_config(
    page_title="Business AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background: #f3f4f6;
        border-left: 3px solid #6366f1;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.75rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_uploaded_files() -> list[str]:
    if not os.path.isdir(UPLOAD_DIR):
        return []
    return sorted(
        f
        for f in os.listdir(UPLOAD_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )


def save_uploaded_file(uploaded_file) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_count" not in st.session_state:
        st.session_state.indexed_count = get_collection_count()


init_session_state()

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Knowledge Base")

    if not GEMINI_API_KEY:
        st.error("Set `GEMINI_API_KEY` in your `.env` file to enable chat.")
    else:
        st.success("API key configured")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, CSV, DOCX, PNG, JPG",
    )

    if uploaded_files:
        if st.button("Save & Index Files", type="primary", use_container_width=True):
            saved = []
            for f in uploaded_files:
                save_uploaded_file(f)
                saved.append(f.name)

            with st.spinner("Building search index..."):
                try:
                    count = rebuild_index()
                    st.session_state.indexed_count = count
                    st.success(f"Indexed {len(saved)} file(s) — {count} chunks total.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()

    files = list_uploaded_files()
    st.markdown(f"**Stored files** ({len(files)})")
    if files:
        for name in files:
            st.caption(f"• {name}")
    else:
        st.caption("No documents yet. Upload files to get started.")

    if st.button("Rebuild Index", use_container_width=True):
        with st.spinner("Rebuilding index from all files..."):
            try:
                count = rebuild_index()
                st.session_state.indexed_count = count
                st.success(f"Index rebuilt — {count} chunks.")
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    chunk_count = st.session_state.indexed_count
    st.markdown(
        f'<div class="stat-card"><div style="font-size:1.5rem;font-weight:700;">{chunk_count}</div>'
        f'<div>Indexed Chunks</div></div>',
        unsafe_allow_html=True,
    )

# Main area
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<p class="main-header">Business AI Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about your uploaded business documents.</p>',
        unsafe_allow_html=True,
    )
with col_status:
    if chunk_count > 0:
        st.info("Ready to answer")
    else:
        st.warning("Upload docs first")

if chunk_count == 0 and not st.session_state.messages:
    st.markdown(
        """
        **Getting started**
        1. Upload PDF, TXT, CSV, or DOCX files in the sidebar
        2. Click **Save & Index Files**
        3. Ask a question in the chat below
        """
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    score = f" (relevance: {src['score']})" if src.get("score") else ""
                    st.markdown(
                        f'<div class="source-box"><strong>{src["file"]}</strong>{score}<br>{src["excerpt"]}</div>',
                        unsafe_allow_html=True,
                    )

if prompt := st.chat_input("Ask anything about your documents..."):
    if not GEMINI_API_KEY:
        st.error("Configure GEMINI_API_KEY before chatting.")
        st.stop()

    if chunk_count == 0:
        st.warning("Upload and index documents before asking questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = query_documents(prompt)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    score = f" (relevance: {src['score']})" if src.get("score") else ""
                    st.markdown(
                        f'<div class="source-box"><strong>{src["file"]}</strong>{score}<br>{src["excerpt"]}</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
