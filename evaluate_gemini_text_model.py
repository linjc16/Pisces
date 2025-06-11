import os
import torch
import pandas as pd
from omegaconf import OmegaConf
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from collections import Counter

from dds.src.model.drug_gemini_text import GeminiTextModel, GeminiTextConfig
from load_cell_features import load_cell_tpm_features
from gemini_test_loader import load_test_data

# Load environment variables
load_dotenv()

# Load test examples and cell line transcriptomics
examples = load_test_data()
tpm_df = load_cell_tpm_features()

# Load SMILES → drug_name mapping
drug_map_df = pd.read_csv("data/drug_smiles.csv")
smiles_to_name = dict(zip(drug_map_df["smiles"], drug_map_df["drug_names"]))

# Initialize model
args = OmegaConf.structured(GeminiTextConfig(embed_dim=512))
model = GeminiTextModel(args)
model.register_classification_head("bclsmlpv2", num_classes=1)
model.eval()

preds, labels = [], []

for ex in tqdm(examples, desc="Evaluating Gemini Text Model"):
    smiles_a = ex["smiles_a"]
    smiles_b = ex["smiles_b"]
    cell_idx = ex["cell_idx"]
    label = int(ex["label"])

    try:
        drug_a = {"smiles": smiles_to_name[smiles_a]}
        drug_b = {"smiles": smiles_to_name[smiles_b]}
    except KeyError as e:
        print(f"⚠️ Skipping pair with missing SMILES: {e}")
        continue

    cell_line = torch.tensor([cell_idx], dtype=torch.long)

    with torch.no_grad():
        out = model(
            drug_a_seq=drug_a,
            drug_b_seq=drug_b,
            drug_a_graph=None,
            drug_b_graph=None,
            cell_line=cell_line,
            classification_head_name="bclsmlpv2"
        )

    prob = torch.sigmoid(out).item()
    pred = 1 if prob >= 0.5 else 0
    preds.append(pred)
    labels.append(label)

# Results
acc = accuracy_score(labels, preds)
print(f"\n Accuracy: {acc:.4f}")
try:
    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"🧠 AUC: {auc:.4f}")
    print(f"🧠 F1 Score: {f1:.4f}")
except ValueError as e:
    print("⚠️ Could not compute AUC/F1:", e)

print("\n Label Distribution:", Counter(labels))
print(" Prediction Distribution:", Counter(preds))
