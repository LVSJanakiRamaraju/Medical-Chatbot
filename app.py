from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = pinecone_api_key or ""
os.environ["GROQ_API_KEY"] = groq_api_key or ""

rag_chain = None


def initialize_rag_chain():
    global rag_chain

    if rag_chain is not None:
        return rag_chain

    if not pinecone_api_key or not groq_api_key:
        print("Pinecone or Groq credentials are missing; chatbot features remain unavailable.")
        return None

    try:
        from src.helper import downloading_hugging_face_embeddings
        from src.prompt import system_prompt
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq
        from langchain_pinecone import PineconeVectorStore

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        embeddings = downloading_hugging_face_embeddings()
        index_name = "test"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)
        question_ans_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        rag_chain = create_retrieval_chain(retriever, question_ans_chain)
    except Exception as exc:
        print(f"Pinecone initialization failed: {exc}")
        rag_chain = None

    return rag_chain


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a message."

    chain = initialize_rag_chain()
    if chain is None:
        return "The chatbot is not ready yet. Please configure PINECONE_API_KEY and GROQ_API_KEY and ensure the vector store is available."

    response = chain.invoke({"input": msg})
    return str(response["answer"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)



