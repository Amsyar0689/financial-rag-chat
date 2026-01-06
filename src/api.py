import os
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "chroma_db"
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        yield
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2
    )

    system_prompt = (
        "You are an expert financial analyst. "
        "Use the retrieved context to answer the question. "
        "If you don't know, say 'I don't know'. "
        "\n\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    
    print("✅ API Ready!")
    yield
    print("🛑 Shutting down...")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

class QueryRequest(BaseModel):
    query: str

async def generate_chat_response(query: str):
    """
    Generator function that streams:
    1. Text tokens (as they are generated)
    2. Source metadata (at the very end)
    """
    if not rag_chain:
        yield json.dumps({"error": "System not ready"}) + "\n"
        return

    stream = rag_chain.stream({"input": query})
    sources_data = []

    for chunk in stream:
        if "answer" in chunk:
            # Yield a JSON line for the frontend to parse
            yield json.dumps({"type": "token", "content": chunk["answer"]}) + "\n"
        
        if "context" in chunk:
            for doc in chunk["context"]:
                sources_data.append({
                    "page": doc.metadata.get("page", 0) + 1,
                    "text": doc.page_content[:300] + "..."
                })

    # After the loop, send the collected sources
    yield json.dumps({"type": "sources", "content": sources_data}) + "\n"

@app.post("/chat")
async def chat(request: QueryRequest):
    return StreamingResponse(
        generate_chat_response(request.query),
        media_type="application/x-ndjson" # Newline Delimited JSON
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)