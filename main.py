import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

print("--- Healix Query Mode (using Local LLM) ---")
print("Type 'exit' or 'quit' to end the program.")

# --- 1. LOAD THE SAVED "MAGIC INDEX" ---
index_path = "faiss_index"
if not os.path.exists(index_path):
    print(f"Error: The FAISS index folder '{index_path}' was not found.")
    exit()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

print("\nLoading the FAISS index...")
db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
print("Index loaded successfully.")

# --- 2. SET UP THE RETRIEVER AND THE LLM ---
# We only need these two components now
retriever = db.as_retriever()
llm = Ollama(model="llama3", temperature=0)

print("Ready to answer questions.\n")

# --- 3. DEFINE PROMPT TEMPLATES ---
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

# --- 4. INTERACTIVE QUESTION LOOP ---
while True:
    query = input("Please enter your medical question: ")

    if query.lower() in ['exit', 'quit']:
        print("Exiting program. Goodbye!")
        break

    print(f"\nSearching for: '{query}'...")

    # Step A: Retrieve relevant documents from our index
    docs = retriever.get_relevant_documents(query)
    if not docs:
        print("Could not find any relevant information for that query. Please try another question.")
        print("\n=======================\n")
        continue

    # Step B: Combine the content of the retrieved documents into a single block of text
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step C: Generate the patient summary
    patient_prompt = patient_template.format(context=context, question=query)
    patient_result = llm.invoke(patient_prompt)
    print("\n--- For the Patient ---")
    print(patient_result)

    print("\n-----------------------\n")

    # Step D: Generate the clinician summary
    clinician_prompt = clinician_template.format(context=context, question=query)
    clinician_result = llm.invoke(clinician_prompt)
    print("--- For the Clinician ---")
    print(clinician_result)
    print("\n=======================\n")
