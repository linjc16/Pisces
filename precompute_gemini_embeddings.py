import pandas as pd
import torch
import google.generativeai as genai
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

df = pd.read_csv("data/drug_smiles.csv")
results = []

model = genai.GenerativeModel(model_name="models/embedding-001")

for _, row in tqdm(df.iterrows(), total=len(df)):
    smiles = row["smiles"]
    try:
        response = model.embed_content(content=smiles, task_type="semantic_similarity")
        embedding = response["embedding"]
        results.append({
            "drug_name": smiles,  
            "embedding": ",".join(map(str, embedding))
        })
    except Exception as e:
        print(f"❌ Failed for {smiles}: {e}")

pd.DataFrame(results).to_csv("data/gemini_embeddings.csv", index=False)
print("✅ Saved embeddings with SMILES as keys.")
