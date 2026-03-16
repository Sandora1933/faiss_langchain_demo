from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(file_path):
    """
    Load documents from a text file.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    return docs


def split_documents(docs, chunk_size=200, chunk_overlap=50):
    """
    Split large documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(docs)
    return chunks