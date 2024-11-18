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

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key is not set. Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

def extract_pdfs_from_zip(zip_file):
    pdf_files = []
    with zipfile.ZipFile(zip_file, 'r') as z:
        for file_name in z.namelist():
            if file_name.lower().endswith('.pdf'):
                with z.open(file_name) as pdf_file:
                    pdf_files.append(io.BytesIO(pdf_file.read()))
    return pdf_files


def get_pdf_text(pdf_docs, folder_paths=None):
    text = ""
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
                                        for page in pdf_reader.pages:
                                            text += page.extract_text()
            else:
                for root, _, files in os.walk(folder_path):
                    for filename in files:
                        if filename.endswith(".pdf"):
                            pdf_path = os.path.join(root, filename)
                            with open(pdf_path, 'rb') as pdf_file:
                                pdf_reader = PdfReader(pdf_file)
                                for page in pdf_reader.pages:
                                    text += page.extract_text()
    if pdf_docs:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks , faiss_index):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(faiss_index)

def get_conversational_chain():
    prompt_template = """
    Answer the question using only the information found within the context.
    If you cannot find the answer, simply state 'answer is not available in the context'

    Context: \n{context}?\n
    Question: \n{question}?\n

    Answer:"""
   
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.write("No relevant documents found.")
            return

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response.get("output_text", "No response generated."))
    except Exception as e:
        st.error(f"Error during user input processing: {e}")

def main():
    st.set_page_config(page_title="Chat with multiple PDFs")
    st.header("Ask the questions from pdf and get the response using Google Gemini Pro!!!")

    user_question = st.text_input("Ask question ?")
   
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        zip_file = st.file_uploader("Upload a ZIP file containing PDF Files", type=["zip"])
        pdf_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        pdf_docs = None

        if zip_file:
            with st.spinner("Extracting PDFs from ZIP..."):
                pdf_docs = extract_pdfs_from_zip(zip_file)
            if not pdf_docs:
                st.error("No PDF files found in the ZIP archive.")
                return

        if pdf_files:
            pdf_docs = pdf_files

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
            with st.spinner("Processing..."):
                text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks,"faiss_index")
                st.success("PDFs processed successfully!")

if __name__ == "__main__":
    main()
