import streamlit as st
import os
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def load_document(uploaded_file):
    """Loads a document based on its file type."""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    with NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        if file_extension == "pdf":
            st.info("📄 Loading PDF document...")
            loader = PyPDFLoader(temp_file_path)
        elif file_extension in ["docx", "doc"]:
            st.info("📝 Loading Word document...")
            loader = UnstructuredWordDocumentLoader(temp_file_path)
        elif file_extension == "txt":
            st.info("📋 Loading text document...")
            loader = TextLoader(temp_file_path, encoding='utf-8')
        else:
            st.error(f"❌ Unsupported file type: {file_extension}")
            st.info("Supported formats: PDF, DOCX, DOC, TXT")
            return None

        documents = loader.load()
        st.success(
            f"✅ Document loaded successfully! Found {len(documents)} pages/sections.")
        return documents

    except Exception as e:
        st.error(f"❌ Error loading document: {str(e)}")
        if "encoding" in str(e).lower():
            st.info("💡 Try saving your text file with UTF-8 encoding.")
        return None
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def chunk_documents(documents):
    """Splits documents into smaller chunks."""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"📊 Document split into {len(chunks)} chunks for processing.")
    return chunks


def create_and_store_embeddings(chunks, embeddings_model, persist_path):
    """Creates embeddings from chunks and stores them in ChromaDB."""
    if not chunks:
        st.warning("⚠️ No chunks to embed. Please upload a valid document.")
        return None

    try:
        st.info("🧠 Creating embeddings and storing in vector database...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_path
        )
        st.success("✅ Document processed and embeddings stored successfully!")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error creating/storing embeddings: {str(e)}")
        if "quota" in str(e).lower():
            st.info(
                "💡 You may have exceeded your API quota. Please check your Google Cloud billing.")
        return None
