import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

INPUT_CSV = "data/drug_smiles.csv"
OUTPUT_CSV = "data/gemini_smiles_embeddings.csv"

df = pd.read_csv(INPUT_CSV)
results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding SMILES"):
    drug_name = row["drug_names"]
    smiles = row["smiles"]

    try:
        res = genai.embed_content(
            model="models/embedding-001",
            content=smiles,
            task_type="semantic_similarity"
        )
        embedding = res["embedding"]
        results.append({
            "drug_name": drug_name,
            "smiles": smiles,  
            "embedding": ",".join(map(str, embedding))
        })
    except Exception as e:
        print(f"❌ Failed to embed {drug_name}: {e}")

if results:
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved SMILES embeddings to {OUTPUT_CSV}")
else:
    print("No embeddings saved. Check model usage or errors above.")
