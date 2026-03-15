import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    load_dotenv()
    loader = TextLoader("data.txt", encoding="utf-8")
    docs = loader.load()
    print(f"Docs: \n{docs}")


if __name__ == "__main__":
    main()


