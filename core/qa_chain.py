from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def create_qa_chain(llm, vectorstore):
    """Creates a RetrievalQA chain with a custom prompt."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Custom prompt template
    template = """You are an AI assistant specialized in legal policy analysis. 
    Context from the document:
    {context}

    Question: {question}

    Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain
