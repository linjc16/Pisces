import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image

# Load your Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Get Gemini vision + embedding models
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# Paths
IMAGE_FOLDER = "molecule_images"
OUTPUT_CSV = "data/gemini_image_text_embeddings.csv"

results = []

# Loop through images
for fname in tqdm(os.listdir(IMAGE_FOLDER), desc="🔍 Describing and Embedding Images"):
    if not fname.endswith(".png"):
        continue

    drug_name = fname.replace(".png", "")
    image_path = os.path.join(IMAGE_FOLDER, fname)

    try:
        # Step 1: Generate description from image
        image = Image.open(image_path)
        response = vision_model.generate_content(["Describe this molecule's structure.", image])
        description = response.text.strip()

        # Step 2: Embed the description
        embedding_res = genai.embed_content(
            model="models/embedding-001",
            content=description,
            task_type="semantic_similarity"
        )
        embedding = embedding_res["embedding"]

        # Step 3: Save result
        results.append({
            "drug_name": drug_name,
            "description": description,
            "embedding": ",".join(map(str, embedding))
        })
    except Exception as e:
        print(f"❌ Failed to process {drug_name}: {e}")

# Save to CSV
if results:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved image text embeddings to {OUTPUT_CSV}")
else:
    print("⚠️ No embeddings saved.")
