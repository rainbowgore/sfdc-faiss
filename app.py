from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

qa_chain = None

def init_qa_engine():
    global qa_chain
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Load FAISS index with allow_dangerous_deserialization
        index_path = "./faiss-lang-pipeline/faiss_index"  # Updated path
        if not os.path.exists(f"{index_path}.faiss"):
            print(f"❌ FAISS index not found at: {index_path}.faiss")
            return False
            
        print("📚 Loading FAISS index...")
        db = FAISS.load_local(
            folder_path=index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS index loaded successfully")
        
        # Initialize QA chain
        print("🔄 Initializing QA chain...")
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever()
        )
        print("✅ QA chain initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing QA engine: {str(e)}")
        return False

def query():
    global qa_chain
    if not qa_chain:
        if not init_qa_engine():
            return jsonify({"error": "Failed to initialize QA engine"}), 500

    try:
        data = request.get_json()
        query_text = data.get("query")
        if not query_text:
            return jsonify({"error": "No query provided"}), 400

        response = qa_chain.run(query_text)
        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query_route():
    return query()

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})