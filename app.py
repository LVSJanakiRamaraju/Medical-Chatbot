from flask import Flask, render_template, jsonify, request
from src.helper import downloading_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GROQ_API_KEY"] = groq_api_key
embeddings = downloading_hugging_face_embeddings()
index_name = "test"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.5
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)
question_ans_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, question_ans_chain)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=["GET","POST"])
def chat():
 msg=request.form["msg"]
 input = msg
 print(input)
 response = rag_chain.invoke({"input": msg})
 print("Response :" , response["answer"])
 return str(response["answer"])


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5001, debug=True)



