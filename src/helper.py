
import os


# Load PDF files from the specified directory
def load_pdf_file(data):
    from langchain_community.document_loaders import PyPDFLoader

    documents = []

    for file in os.listdir(data):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data, file)

            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                documents.extend(docs)

            except Exception as e:
                print(f"Error loading {file}: {e}")

    return documents


# Split the loaded documents into smaller chunks
def text_splitter(extracted_data):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(extracted_data)
    return split_docs


# Create embeddings for the split documents using HuggingFaceEmbeddings
def downloading_hugging_face_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings