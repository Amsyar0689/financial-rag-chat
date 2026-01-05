import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PDF_PATH = "data/apple_10k_2025.pdf"
DB_PATH = "chroma_db"

def ingest_documents():
    print("STARTING INGESTION...")
    
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks.")

    print("Sending to Google Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )

    print(f"DONE! Vector Database successfully built .")

if __name__ == "__main__":
    ingest_documents()