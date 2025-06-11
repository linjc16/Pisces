import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

# Load Gemini API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")

# Input/output paths
input_path = "data/drug_smiles.csv"
output_path = "data/gemini_descriptions.csv"

df = pd.read_csv(input_path)
results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating molecular descriptions"):
    drug_name = row["drug_names"]
    smiles = row["smiles"]

    prompt = f"Describe the molecular structure of this SMILES string: {smiles}"

    try:
        res = model.generate_content(prompt)
        description = res.text or res.candidates[0].content.parts[0].text
        results.append({
            "drug_name": drug_name,
            "smiles": smiles,
            "description": description.strip()
        })
    except Exception as e:
        print(f"❌ Failed to process {drug_name}: {e}")

pd.DataFrame(results).to_csv(output_path, index=False)
print(f"\n✅ Descriptions saved to: {output_path}")
