import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.utils import build_prompt
from core.llm import create_llm
from core.prepare_data import load_documents, split_documents
from core.retriever import create_retriever


def get_response(retriever, query, chat_history):
    matching_docs = retriever.invoke(query)
    print(f"Matching Docs: \n{matching_docs}")

    context = "\n".join([doc.page_content for doc in matching_docs])
    prompt_template = build_prompt()
    llm = create_llm()

    chain = prompt_template | llm
    response = chain.invoke({
        "query": query,
        "context": context,
        "chat_history": chat_history
    })

    return response

def main():
    load_dotenv()
    docs = load_documents("data/data.txt")
    print(f"Docs: \n{docs}")

    chunks = split_documents(docs)
    retriever = create_retriever(chunks)

    query = "Can I come to the store on Sunday?"
    chat_history = []
    response = get_response(retriever, query, chat_history)
    print(f"Response: {response.content}")
    

if __name__ == "__main__":
    main()


