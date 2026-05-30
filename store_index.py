# inorder to store extra information folr future purpose in pincone database.
from src.helper import (
    load_pdf_file,
    text_splitter,
    downloading_hugging_face_embeddings
)
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = pinecone_api_key

extracted_data=load_pdf_file(data='data/')
text_chunks=text_splitter(extracted_data)
embeddings=downloading_hugging_face_embeddings()
 
from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index = pc.Index("test")

index.delete(delete_all=True)

print("All vectors deleted")

from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
   index_name="test",
    embedding=embeddings
)
    



