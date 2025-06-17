from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import os
import re

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path="faiss-lang-pipeline/.env")

app = Flask(__name__)
CORS(app)

# Initialize FAISS and QA chain
embeddings = OpenAIEmbeddings()
db = None
qa_chain = None
case_dict = None

def init_qa_engine():
    global db, qa_chain, case_dict
    try:
        # Load CSV data for fallback
        csv_path = "final-csv/enriched_cases.csv"
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        case_dict = df.set_index("CaseId").to_dict(orient="index")

        # Load FAISS index
        db = FAISS.load_local("faiss-lang-pipeline/faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
        return True
    except Exception as e:
        print(f"❌ Error initializing QA engine: {e}")
        return False

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route("/enrich-cases", methods=["POST"])
def enrich_cases():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({"error": "No selected file or invalid format"}), 400

    try:
        # Read the CSV file
        df = pd.read_csv(BytesIO(file.read()))
        
        # Process the file (add your enrichment logic here)
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='enriched_cases.csv'
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query_cases():
    if not qa_chain:
        if not init_qa_engine():
            return jsonify({"error": "QA engine initialization failed"}), 500

    try:
        data = request.json
        query = data.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Get answer from QA chain
        answer = qa_chain.invoke({"query": query})
        response = answer.get('result', '') if isinstance(answer, dict) else answer

        # Fallback logic with intent detection
        if "I don't have information" in response:
            case_id_match = re.search(r"\b(500gK\w+)\b", query)
            if case_id_match:
                case_id = case_id_match.group(1)
                if case_dict and case_id in case_dict:
                    case = case_dict[case_id]
                    
                    # Field mapping based on query intent
                    field_mapping = {
                        "urgency": "Urgency__c",
                        "sentiment": "Sentiment__c",
                        "summary": "Summary__c",
                        "solution": "Suggested_Solution__c",
                        "status": "Status",
                        "priority": "Priority",
                        "description": "Description",
                    }

                    # Detect intent from query
                    query_lower = query.lower()
                    field = None
                    
                    for keyword, field_name in field_mapping.items():
                        if keyword in query_lower:
                            field = field_name
                            break
                    
                    # Default to Suggested_Solution__c if no specific intent found
                    if not field:
                        field = "Suggested_Solution__c"
                        
                    value = case.get(field, "No value found for this field.")
                    
                    return jsonify({
                        "answer": f"Found for case {case_id}:\n{field}: {value}",
                        "source": "fallback",
                        "field": field
                    })
                else:
                    return jsonify({
                        "answer": f"Case {case_id} not found in the database.",
                        "source": "fallback"
                    })
            else:
                return jsonify({
                    "answer": "No recognizable Case ID found in query.",
                    "source": "fallback"
                })

        # Return original response if fallback wasn't needed
        return jsonify({"answer": response, "source": "qa_chain"})

    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)