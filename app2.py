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
                    pdf_files.append((file_name, io.BytesIO(pdf_file.read())))
    return pdf_files

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
                                texts.append(text)
                                document_sources.append(pdf_path)
    if pdf_docs:
        for pdf_name, pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            texts.append(text)
            document_sources.append(pdf_name)

    return texts, document_sources

def get_text_chunks(texts, document_sources):
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = []
    chunk_sources = []
    for text, source in zip(texts, document_sources):
        chunks = text_splitter.split_text(text)
        text_chunks.extend(chunks)
        chunk_sources.extend([source] * len(chunks))
    return text_chunks, chunk_sources

def get_vector_store(text_chunks, chunk_sources, faiss_index):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
  
    metadatas = [{"source": chunk_sources[i]} for i in range(len(text_chunks))]
    
   
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(faiss_index)
    
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

def main():
    st.title("Ask the questions from PDF and get the response using Google Gemini Pro!!!")
    
    uploaded_files = st.file_uploader("Upload PDF or ZIP files", type=["pdf", "zip"], accept_multiple_files=True)
    
    if uploaded_files:
        pdf_docs = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.zip'):
                pdf_docs.extend(extract_pdfs_from_zip(uploaded_file))
            elif uploaded_file.name.lower().endswith('.pdf'):
                pdf_docs.append((uploaded_file.name, uploaded_file))
        
        texts, document_sources = get_pdf_text(pdf_docs)
        text_chunks, chunk_sources = get_text_chunks(texts, document_sources)
        get_vector_store(text_chunks, chunk_sources, "faiss_index")
        
        user_question = st.text_input("Ask a question about the PDF content:")
        
        if user_question:
            user_input(user_question)

if __name__ == "__main__":
    main()




