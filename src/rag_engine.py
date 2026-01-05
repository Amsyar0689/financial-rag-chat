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
    # 1. Verification
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at '{DB_PATH}'.")
        print("Please run 'python src/ingest.py' first!")
        return

    print("🤖 Initializing Financial Analyst AI (Gemini 1.5 Flash)...")

    try:
        # 2. Load the Knowledge Base (Embeddings)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        
        # Setup the Retriever (The Search Engine)
        # k=5 means we fetch the top 5 most relevant chunks
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})

        # 3. Setup the Brain (LLM)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,      # 0 = Strict facts, 1 = Creative
            max_retries=2,
        )

        # 4. Define the "System Instructions" (The Prompt)
        system_prompt = (
            "You are an expert financial analyst. "
            "Use the retrieved context below to answer the user's question. "
            "If the answer is not in the context, say 'I don't know' or 'The document doesn't mention this'. "
            "Keep your answer professional and concise."
            "\n\n"
            "Context:\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # 5. Build the Modern Chain (LCEL Architecture)
        # Step A: "Stuff" the docs into the prompt
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        # Step B: Connect the retriever to the Q&A chain
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        print("System Ready! Ask about the Apple 10-K (Type 'exit' to quit).")
        print("-" * 60)

        # 6. The Chat Loop
        while True:
            query = input("\nStart chatting... \nYou: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            print("Searching & Thinking...", end="\r")
            
            try:
                start_time = time.time()
                
                # Run the RAG pipeline
                response = rag_chain.invoke({"input": query})
                
                end_time = time.time()
                elapsed = end_time - start_time

                # Extract answer and sources
                answer = response["answer"]
                sources = response["context"]

                print(f"(Response time: {elapsed:.2f}s)")
                print(f"\nAI: {answer}")
                
                print("\nSources Used:")
                if not sources:
                    print("    - No specific sources found in context.")
                else:
                    for i, doc in enumerate(sources[:3]): # Show top 3 sources
                        # Page numbers in PyPDF start at 0, so we add 1
                        page_num = int(doc.metadata.get('page', 0)) + 1
                        # Clean up text for display
                        snippet = doc.page_content[:100].replace('\n', ' ')
                        print(f"      - Page {page_num}: {snippet}...")
                    
            except Exception as e:
                print(f"\nError during processing: {e}")

    except Exception as e:
        print(f"\nCritical System Error: {e}")

if __name__ == "__main__":
    start_chat()