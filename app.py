from dotenv import load_dotenv
import streamlit as st
import os
import shutil
from pathlib import Path
from utils.document_processor import load_document, chunk_documents, create_and_store_embeddings
from utils.llm_manager import initialize_gemini_models
from core.qa_chain import create_qa_chain
from config import CHROMA_DB_PATH

# Streamlit page configuration
st.set_page_config(
    page_title="Legal Policy Reader AI Agent",
    layout="wide",
    page_icon="📄"
)

# Load environment variables and check API key
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY", "")
if not api_key or not api_key.strip() or not api_key.startswith("AIza"):
    st.error(
        "🚨 Invalid or missing `GOOGLE_API_KEY`. Please set it in your environment variables.")
    st.stop()

# --- Initialize LLM and Embeddings using st.session_state ---
if 'llm' not in st.session_state or 'embeddings' not in st.session_state:
    with st.spinner("Initializing Gemini models..."):
        st.session_state.llm, st.session_state.embeddings = initialize_gemini_models(
            api_key)

llm = st.session_state.llm
embeddings = st.session_state.embeddings

# --- Main Streamlit UI ---
st.title("📄 Legal Policy Reader AI Agent")
st.markdown("""
🤖 **Upload your policy document and ask questions about its content**

Supported formats: PDF, DOCX, DOC, TXT
""")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown("""
    1. Upload your document
    2. Wait for processing
    3. Ask questions about the content
    4. Get AI-powered answers with sources
    """)

    st.header("🔧 System Status")
    st.success("✅ Gemini Models: Ready")
    if 'vectorstore' in st.session_state:
        st.success("✅ Document: Loaded")
    else:
        st.info("⏳ Document: Not loaded")

# File Upload Section
st.header("📤 Upload Document")
uploaded_file = st.file_uploader(
    "Choose a policy document",
    type=["pdf", "docx", "doc", "txt"],
    help="Upload PDF, Word document, or text file"
)

if uploaded_file:
    st.info(f"📁 Processing '{uploaded_file.name}'...")

    # Load document
    documents = load_document(uploaded_file)

    if documents:
        with st.spinner("✂️ Chunking document..."):
            chunks = chunk_documents(documents)

        if chunks:
            with st.spinner("🧠 Creating embeddings..."):
                vectorstore = create_and_store_embeddings(
                    chunks, embeddings, CHROMA_DB_PATH)

            if vectorstore:
                st.session_state['vectorstore'] = vectorstore
                st.session_state['document_name'] = uploaded_file.name
                st.balloons()
            else:
                st.error("❌ Failed to prepare document for querying.")
    else:
        st.error(
            "❌ Could not load the document. Please check the file format and content.")

# Query Section
st.header("❓ Ask a Question")

if 'document_name' in st.session_state:
    st.info(f"📋 Current document: **{st.session_state['document_name']}**")

# Sample questions
with st.expander("💡 Sample Questions"):
    sample_questions = [
        "What is the company's remote work policy?",
        "Are there any restrictions on data usage?",
        "What are the consequences of policy violations?",
        "Does the policy mention confidentiality agreements?",
        "What are the employee benefits mentioned?"
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{hash(q)}"):
            st.session_state['query_input'] = q

user_query = st.text_area(
    "Enter your question about the policy document:",
    height=100,
    value=st.session_state.get('query_input', ''),
    placeholder="e.g., What is the company's vacation policy?"
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("🔍 Get Answer", type="primary")
with col2:
    if st.button("🗑️ Clear Query"):
        st.session_state['query_input'] = ''
        st.rerun()

if ask_button:
    if 'vectorstore' not in st.session_state or st.session_state['vectorstore'] is None:
        st.warning("⚠️ Please upload and process a document first.")
    elif not user_query.strip():  # type: ignore
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤔 Analyzing document and generating answer..."):
            try:
                qa_chain = create_qa_chain(
                    llm, st.session_state['vectorstore'])
                result = qa_chain({"query": user_query})

                # Display answer
                st.subheader("🎯 Answer:")
                st.write(result["result"])

                # Display source documents
                if result["source_documents"]:
                    with st.expander("📚 Source Documents (Click to expand)", expanded=False):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**📄 Relevant Chunk {i+1}:**")
                            st.text_area(
                                f"Content {i+1}:",
                                doc.page_content,
                                height=150,
                                key=f"chunk_{i}"
                            )
                            if doc.metadata:
                                st.json(doc.metadata)
                            st.divider()

            except Exception as e:
                st.error(
                    f"❌ An error occurred during query processing: {str(e)}")
                st.info("💡 Try:")
                st.write("- Re-uploading the document")
                st.write("- Rephrasing your question")
                st.write("- Checking your internet connection")

# --- Footer and utilities ---
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ Clear Document Data"):
        try:
            if Path(CHROMA_DB_PATH).exists():
                shutil.rmtree(CHROMA_DB_PATH)

            # Clear session state
            for key in ['vectorstore', 'document_name', 'query_input']:
                if key in st.session_state:
                    del st.session_state[key]

            st.success(
                "✅ Document data cleared! Upload a new document to continue.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error clearing data: {str(e)}")

with col2:
    if st.button("🔄 Reset Application"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Application reset! Please refresh the page.")

st.markdown("---")
st.caption("🚀 Built with Streamlit, LangChain, Google Gemini, and ChromaDB")

# Debug information (only show in development)
if st.checkbox("🔧 Show Debug Info"):
    st.subheader("Debug Information")
    st.write("Session State Keys:", list(st.session_state.keys()))
    st.write("Environment Variables:", [
             k for k in os.environ.keys() if 'GOOGLE' in k])
    if Path(CHROMA_DB_PATH).exists():
        st.write("ChromaDB Directory exists: ✅")
    else:
        st.write("ChromaDB Directory exists: ❌")
