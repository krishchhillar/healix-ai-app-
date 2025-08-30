from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os
# This is new! We need to allow our frontend to talk to our backend.
from fastapi.middleware.cors import CORSMiddleware

# --- SETUP: LOAD MODELS ON STARTUP ---
state = {}

def load_models_and_index():
    print("Loading models and FAISS index...")
    index_path = "faiss_index"
    if not os.path.exists(index_path):
        raise RuntimeError("FAISS index not found. Please run the initial setup script first.")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    state["retriever"] = db.as_retriever()
    state["llm"] = Ollama(model="llama3", temperature=0)
    print("Models and index loaded successfully.")

# --- API DEFINITION ---
app = FastAPI(
    title="Healix API",
    description="API for generating patient and clinician medical summaries.",
    version="1.0.0",
)

# --- NEW: Add CORS Middleware ---
# This allows your frontend (even if it's just an HTML file) to make requests to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- MODIFIED: Input model to match the frontend ---
class Query(BaseModel):
    message: str
    audience: str

@app.on_event("startup")
async def startup_event():
    load_models_and_index()

# --- MODIFIED: Endpoint changed from /summarize to /chat ---
@app.post("/chat")
async def chat(query: Query):
    user_query = query.message
    audience = query.audience
    print(f"Received query: '{user_query}' for audience: '{audience}'")

    docs = state["retriever"].get_relevant_documents(user_query)
    if not docs:
        return {"response": "I'm sorry, I couldn't find any relevant information for that topic."}
    
    context = "\n\n".join([doc.page_content for doc in docs])

    summary = ""
    # --- MODIFIED: Logic to generate only the requested summary ---
    if audience == 'patient':
        patient_template = f"""You are a friendly and empathetic nurse. Based ONLY on the medical information below, explain the answer to the patient's question in simple, easy-to-understand terms. Medical Information: {context}. Patient's Question: {user_query}. Your simple explanation: """
        summary = state["llm"].invoke(patient_template)
    elif audience == 'doctor':
        clinician_template = f"""You are a medical professional. Based ONLY on the context below, provide a concise, technical summary for a colleague. Context: {context}. Question: {user_query}. Your technical summary: """
        summary = state["llm"].invoke(clinician_template)
    else:
        return {"response": "Invalid audience specified."}
        
    print("Successfully generated summary.")
    
    # --- MODIFIED: Return structure to match frontend's expectation ---
    return {"response": summary}

@app.get("/")
def read_root():
    return {"status": "Healix API is running."}

