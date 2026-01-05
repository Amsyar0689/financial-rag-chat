import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API Key
load_dotenv()

# Configuration
DB_PATH = "chroma_db"

def start_chat():
    # 1. Check if DB exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at '{DB_PATH}'. Run ingest.py first!")
        return

    print("Initializing Financial Analyst AI (Gemini 1.5 Flash)...")

    # 2. Load the Knowledge Base
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # 3. Setup the Brain (LLM)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        max_retries=2
    )

    # 4. Create the System Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an expert financial analyst. Use the following context to answer the question.
    If the answer is not in the context, say "I don't know" or "The document doesn't mention this."
    Keep your answers concise and professional.
    
    Context:
    {context}
    
    Question:
    {input}
    """)

    # 5. Build the Modern Chain (LCEL)
    # First, a chain that processes the documents ("stuff" them into context)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Second, the retrieval chain that fetches docs and passes them to the doc chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("System Ready! Ask about the Apple 10-K (Type 'exit' to quit).")
    print("-" * 60)

    while True:
        query = input("\nStart chatting... \nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        print("   🔍 Searching & Thinking...", end="\r")
        
        try:
            start_time = time.time()
            
            # Invoke the chain
            response = retrieval_chain.invoke({"input": query})
            
            end_time = time.time()
            
            answer = response["answer"]
            # In the new chain, sources are in "context" key
            sources = response["context"]

            print(f"   ⏱️  (Response time: {end_time - start_time:.2f}s)")
            print(f"\nAI: {answer}")
            print("\n   📄 Sources Used:")
            for i, doc in enumerate(sources[:3]): # Show top 3 sources
                # +1 because pages start at 0
                page_num = int(doc.metadata.get('page', 0)) + 1 
                # Clean up newlines for cleaner display
                snippet = doc.page_content[:100].replace('\n', ' ')
                print(f"      - Page {page_num}: {snippet}...")
                
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    start_chat()