import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Path to your enriched CSV
input_csv = "/Users/noasasson/Dev-projects/sfdc-aqua/final-csv/enriched_cases.csv"

# Load the data
df = pd.read_csv(input_csv)

# Combine fields into text chunks
def row_to_text(row):
    return "\n".join([
        f"CaseId: {row['CaseId']}",
        f"Summary: {row['Summary__c']}",
        f"Sentiment: {row['Sentiment__c']}",
        f"Sentiment Reason: {row['Sentiment_Reason__c']}",
        f"Suggested Solution: {row['Suggested_Solution__c']}",
        f"Urgency: {row['Urgency__c']}"
    ])

docs = [
    Document(
        page_content=row_to_text(row),
        metadata={"CaseId": row['CaseId']}
    )
    for _, row in df.iterrows()
]

# Split into manageable chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# Embed and build FAISS index
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(split_docs, embeddings)

# Save locally
db.save_local("faiss_index")
print("✅ FAISS index saved to: faiss_index")