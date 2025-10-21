import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Healix",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM STYLING (FINAL FIX) ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #F0F2F6;
    }
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Main title and subtitle text color */
    .stApp h1, .stApp .stMarkdown p {
        color: #31333F;
    }
    /* Text input box */
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        color: #31333F;
    }
    /* Button styling */
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem 1.5rem;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    /* Response cards */
    .response-card {
        background-color: #FFFFFF;
        border-radius: 0.74rem;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #E0E0E0;
    }
    .response-card h3 {
        color: #007BFF;
        margin-top: 0;
    }
    /* --- THE NEW, MORE RELIABLE FIX --- */
    /* This class is now applied directly to the AI's output text */
    .result-text {
        color: #31333F !important; /* Force dark text color */
        white-space: pre-wrap;      /* Ensures line breaks are respected */
    }
</style>
""", unsafe_allow_html=True)


# --- CACHING MODELS AND DATA ---
@st.cache_resource
def load_models_and_index():
    """Loads the FAISS index, embedding model, and the LLM."""
    index_path = "faiss_index"
    if not os.path.exists(index_path):
        st.error("FAISS index not found. Please run the initial setup script first.")
        st.stop()
    
    # Load the components
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    llm = Ollama(model="llama3", temperature=0)
    
    return retriever, llm

# --- PROMPT TEMPLATES ---
patient_template = """
You are a friendly and empathetic nurse. Based ONLY on the medical information below,
explain the answer to the patient's question in simple, easy-to-understand terms.
Do not use any complex medical jargon.

Medical Information:
{context}

Patient's Question: {question}

Your simple explanation: """

clinician_template = """
You are a medical professional. Based ONLY on the context below, provide a concise,
technical summary for a colleague. Use appropriate medical terminology.

Context:
{context}

Question: {question}

Your technical summary: """


# --- APP LAYOUT AND LOGIC ---
st.title("⚕️ Healix: Medical Summarizer")
st.markdown("Enter a medical question or topic to receive two summaries: one simplified for patients and one technical for clinicians.")

# Load everything once and cache it
retriever, llm = load_models_and_index()

# User input
user_query = st.text_input("Enter your medical question:", placeholder="e.g., What is hypertension?")

if st.button("Generate Summaries"):
    if user_query:
        with st.spinner("Searching for information and generating summaries... Please wait."):
            # 1. Retrieve context
            docs = retriever.get_relevant_documents(user_query)
            if not docs:
                st.warning("Could not find relevant information for that query. Please try another question.")
            else:
                context = "\n\n".join([doc.page_content for doc in docs])

                # 2. Generate summaries in two columns
                col1, col2 = st.columns(2)

                # Patient Summary
                with col1:
                    st.markdown("<div class='response-card'><h3>For the Patient</h3>", unsafe_allow_html=True)
                    patient_prompt = patient_template.format(context=context, question=user_query)
                    patient_result = llm.invoke(patient_prompt)
                    # Use markdown with our custom class instead of st.write
                    st.markdown(f"<div class='result-text'>{patient_result}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Clinician Summary
                with col2:
                    st.markdown("<div class='response-card'><h3>For the Clinician</h3>", unsafe_allow_html=True)
                    clinician_prompt = clinician_template.format(context=context, question=user_query)
                    clinician_result = llm.invoke(clinician_prompt)
                    # Use markdown with our custom class instead of st.write
                    st.markdown(f"<div class='result-text'>{clinician_result}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Please enter a question.")



