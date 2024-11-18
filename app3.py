import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import zipfile
import tempfile
import io

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key is not set. Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Function to extract PDFs from a zip file
def extract_pdfs_from_zip(zip_file):
    pdf_files = []
    with zipfile.ZipFile(zip_file, 'r') as z:
        for file_name in z.namelist():
            if file_name.lower().endswith('.pdf'):
                with z.open(file_name) as pdf_file:
                    pdf_files.append((file_name, io.BytesIO(pdf_file.read())))
    return pdf_files

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs, folder_paths=None):
    texts = []
    document_sources = []
    if folder_paths:
        for folder_path in folder_paths:
            if folder_path.endswith(".zip"):
                with zipfile.ZipFile(folder_path, 'r') as zip_ref:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_ref.extractall(temp_dir)
                        for root, _, files in os.walk(temp_dir):
                            for filename in files:
                                if filename.endswith(".pdf"):
                                    pdf_path = os.path.join(root, filename)
                                    with open(pdf_path, 'rb') as pdf_file:
                                        pdf_reader = PdfReader(pdf_file)
                                        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                                        if text.strip():  # Avoid empty text
                                            texts.append(text)
                                            document_sources.append(pdf_path)
            else:
                for root, _, files in os.walk(folder_path):
                    for filename in files:
                        if filename.endswith(".pdf"):
                            pdf_path = os.path.join(root, filename)
                            with open(pdf_path, 'rb') as pdf_file:
                                pdf_reader = PdfReader(pdf_file)
                                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                                if text.strip():  # Avoid empty text
                                    texts.append(text)
                                    document_sources.append(pdf_path)
    if pdf_docs:
        for pdf_name, pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            if text.strip():  # Avoid empty text
                texts.append(text)
                document_sources.append(pdf_name)

    return texts, document_sources

# Function to split text into chunks
def get_text_chunks(texts, document_sources):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = []
    chunk_sources = []
    for text, source in zip(texts, document_sources):
        chunks = text_splitter.split_text(text)
        text_chunks.extend(chunks)
        chunk_sources.extend([source] * len(chunks))
    return text_chunks, chunk_sources

# Function to create and save the FAISS vector store
def get_vector_store(text_chunks, chunk_sources, faiss_index):
    if not text_chunks:
        st.warning("No text chunks to process.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Debug: Check embeddings
    print("Generated embeddings for:", len(text_chunks), "chunks.")
    
    metadatas = [{"source": chunk_sources[i]} for i in range(len(text_chunks))]
    
    # Debug: Check chunk sources and metadata
    print("Chunk sources:", chunk_sources[:5])  # First 5 chunk sources
    print("Metadatas:", metadatas[:5])  # First 5 metadatas
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(faiss_index)

# Function to set up the question answering chain
def get_conversational_chain():
    prompt_template = """
    Answer the question using only the information found within the context.
    If you cannot find the answer, simply state 'answer is not available in the context'.

    Context: \n{context}\n
    Question: \n{question}\n

    Answer:"""
   
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to handle user input and provide answers
def user_input(user_question):
    try:
        print("Processing user question...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.write("No relevant documents found.")
            return

        chain = get_conversational_chain()
        for doc in docs:
            context = doc.page_content
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown source')
            
            response = chain.run(input_documents=[doc], context=context, question=user_question)
            response = response.strip()
            if response and response.lower() != "answer is not available in the context":
                st.write(f"Answer: {response}\nSource: {source}")
                return

        st.write("Answer is not available in the context")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main function to tie everything together
def main():
    st.title("Ask the questions from PDF and get the response using Google Gemini Pro!!!")
    
    uploaded_file = st.file_uploader("Upload a PDF files", type=["pdf", "zip"])
    
    if uploaded_file is not None:
        # Process uploaded zip file or single PDF
        if uploaded_file.name.endswith(".zip"):
            pdf_docs = extract_pdfs_from_zip(uploaded_file)
        else:
            pdf_docs = [(uploaded_file.name, uploaded_file)]
        
        texts, document_sources = get_pdf_text(pdf_docs)
        if not texts:
            st.warning("No text extracted from the PDFs.")
            return

        text_chunks, chunk_sources = get_text_chunks(texts, document_sources)
        if not text_chunks:
            st.warning("No valid text chunks to process.")
            return

        get_vector_store(text_chunks, chunk_sources, "faiss_index")
        
        # User input section for querying
        user_question = st.text_input("Ask a question about the PDF content:")
        
        if user_question:
            user_input(user_question)

if __name__ == "__main__":
    main()

