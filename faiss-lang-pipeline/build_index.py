# build_index.py

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

CSV_PATH = "final-csv/enriched_cases.csv"
INDEX_DIR = "faiss-lang-pipeline/faiss_index"

def build_index():
    print("🔄 Loading enriched cases CSV...")
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    texts = df["Conversation"].tolist()
    metadatas = df[["CaseId"]].to_dict(orient="records")

    print("🧠 Creating embeddings...")
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    print(f"💾 Saving FAISS index to {INDEX_DIR}...")
    db.save_local(INDEX_DIR)
    print("✅ Done!")

if __name__ == "__main__":
    build_index()