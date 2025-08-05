import streamlit as st
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai


def initialize_gemini_models(api_key):
    """Initializes and returns the Gemini LLM and Embeddings model."""
    try:
        st.info("🔧 Initializing Gemini models...")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        llm_instance = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            api_key=api_key
        )

        embeddings_instance = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        st.success("✅ Gemini models initialized successfully!")
        return llm_instance, embeddings_instance

    except Exception as e:
        st.error(f"❌ Error initializing Gemini models: {str(e)}")
        st.info("Please check:")
        st.write("- Your API key is valid and active")
        st.write("- You have internet connection")
        st.write("- The Gemini API is accessible from your location")

        st.info("🔄 Trying alternative initialization...")
        try:
            genai.configure(api_key=api_key) # type: ignore

            llm_instance = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.1
            )
            embeddings_instance = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            st.success("✅ Alternative initialization successful!")
            return llm_instance, embeddings_instance
        except Exception as e2:
            st.error(f"❌ Alternative method also failed: {str(e2)}")
            st.stop()
