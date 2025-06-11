import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Input folder and output file
image_folder = "molecule_images"
output_csv = "data/gemini_image_embeddings.csv"

# Set up the Gemini multimodal model
vision_model = genai.GenerativeModel("gemini-1.5-pro-vision")

# Store results
results = []

# Iterate over each PNG file
for filename in tqdm(os.listdir(image_folder), desc="Embedding molecule images"):
    if filename.endswith(".png"):
        drug_name = os.path.splitext(filename)[0]
        image_path = os.path.join(image_folder, filename)

        try:
            with Image.open(image_path) as img:
                response = vision_model.embed_image(img)
                embedding = response["embedding"]
                results.append({
                    "drug_name": drug_name,
                    "embedding": ",".join(map(str, embedding))
                })
        except Exception as e:
            print(f"❌ Failed to embed {drug_name}: {e}")

# Save to CSV
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved image embeddings to {output_csv}")
else:
    print("No embeddings saved.")
