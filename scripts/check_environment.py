"""Verify the RAG dependencies without making paid API requests."""

import sys
from importlib.metadata import version

import langchain
import streamlit
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    assert sys.version_info[:2] == (3, 11), sys.version
    load_dotenv()
    splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=5)
    chunks = splitter.split_documents([Document(page_content="RAG chatbot environment. " * 5)])
    assert len(chunks) > 1
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    assert prompt.invoke({"question": "Hello"}).to_messages()
    # A placeholder validates construction without requiring a real API key.
    ChatOpenAI(api_key="environment-check-placeholder")
    OpenAIEmbeddings(api_key="environment-check-placeholder")
    print(f"Python {sys.version.split()[0]}")
    for package in (
        "langchain", "langchain-openai", "langchain-text-splitters",
        "streamlit", "python-dotenv",
    ):
        print(f"{package}=={version(package)}")
    print("PASS: imports, document splitting, prompt formatting, OpenAI client construction")


if __name__ == "__main__":
    main()
