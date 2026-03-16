import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prompts import PREFIX, QUESTION_PROMPT


def get_response(retriever, query, chat_history):
    matching_docs = retriever.invoke(query)
    print(f"Matching Docs: \n{matching_docs}")
    context = "\n".join([doc.page_content for doc in matching_docs])
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", PREFIX + "\n" + QUESTION_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{query}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    chain = prompt_template | llm
    response = chain.invoke({
        "query": query,
        "context": context,
        "chat_history": chat_history
    })

    return response

def main():
    load_dotenv()
    loader = TextLoader("data.txt", encoding="utf-8")
    docs = loader.load()
    print(f"Docs: \n{docs}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever()

    query = "Can I come to the store on Sunday?"
    chat_history = []
    response = get_response(retriever, query, chat_history)
    print(f"Response: {response.content}")
    


if __name__ == "__main__":
    main()


