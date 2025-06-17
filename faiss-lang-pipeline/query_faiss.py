from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI  # Updated import
import os
import pandas as pd
import re

from dotenv import load_dotenv
load_dotenv()

# Check for OpenAI key
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("❌ Missing OPENAI_API_KEY. Please set it in your environment variables.")

# Load the CSV file first
csv_path = "/Users/noasasson/Dev-projects/sfdc-aqua/final-csv/enriched_cases.csv"
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
case_dict = df.set_index("CaseId").to_dict(orient="index")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create or load FAISS index
index_path = "faiss_index"
if os.path.exists(f"{index_path}.faiss"):
    print("📚 Loading existing FAISS index...")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("🔨 Creating new FAISS index...")
    texts = df['text'].tolist()  # Make sure 'text' matches your CSV column name
    db = FAISS.from_texts(texts, embeddings)
    db.save_local(index_path)
    print("✅ FAISS index created and saved!")

# Set up the QA chain
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Interactive prompt
print("\n💬 Ask me something about your support cases. Type 'exit' to bail.\n")

while True:
    try:
        query = input("🧠 Your Query: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting. Bye, human.")
            break
        if not query:
            print("🙄 You didn't type anything. Try again.")
            continue
        response = qa_chain.run(query)
        if "I don't have information" in response:
            case_id_match = re.search(r"\b(500gK\w+)\b", query)
            if case_id_match:
                case_id = case_id_match.group(1)
                if case_id in case_dict:
                    case = case_dict[case_id]
                    print("📝 Found the case directly:")
                    print(f"Suggested_Solution__c: {case['Suggested_Solution__c']}")
                else:
                    print(f"⚠️ Case {case_id} not found in FAISS or in the data.")
            else:
                print("⚠️ No recognizable Case ID found in query.")
        else:
            print(f"📄 Answer:\n{response}")
    except Exception as e:
        print(f"💥 An error occurred: {e}")