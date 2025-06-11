import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

INPUT_CSV = "data/gemini_descriptions.csv"
OUTPUT_CSV = "data/gemini_text_embeddings.csv"

df = pd.read_csv(INPUT_CSV)
results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding descriptions"):
    drug_name = row["drug_name"]
    description = row["description"]

    try:
        res = genai.embed_content(
            model="models/embedding-001",
            content=description,
            task_type="semantic_similarity"
        )
        embedding = res["embedding"]
        results.append({
            "drug_name": drug_name,
            "embedding": ",".join(map(str, embedding))
        })
    except Exception as e:
        print(f"❌ Failed to embed {drug_name}: {e}")

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved embeddings to {OUTPUT_CSV}")
