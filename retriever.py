from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_retriever(chunks):
    """
    Create a retriever from document chunks.
    """

    # Embedding model that converts text into numerical vectors
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build FAISS vector index from the chunk embeddings
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create a retriever that performs similarity search over the vector store
    retriever = vectorstore.as_retriever()

    return retriever