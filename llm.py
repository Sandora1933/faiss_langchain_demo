import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
API_KEY = os.getenv("OPENAI_API_KEY")

def create_llm():
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY
    )