import os
import torch
import pandas as pd
from omegaconf import OmegaConf
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from collections import Counter

from dds.src.model.drug_gemini import GeminiModel, GeminiConfig  
from gemini_test_loader import load_test_data
from load_cell_features import load_cell_tpm_features

# Load environment variables
load_dotenv()

# Load test data and cell features
examples = load_test_data()
tpm_df = load_cell_tpm_features()

# Initialize Gemini SMILES model
args = OmegaConf.structured(GeminiConfig(gnn_embed_dim=768, embedding_source="smiles"))
model = GeminiModel(args)
model.register_classification_head("bclsmlpv2", num_classes=1)
model.eval()

preds, labels = [], []

for ex in tqdm(examples, desc="Evaluating Gemini SMILES Model"):
    drug_a = {"smiles": ex["smiles_a"]}
    drug_b = {"smiles": ex["smiles_b"]}
    cell_line = torch.tensor([ex["cell_idx"]], dtype=torch.long)

    with torch.no_grad():
        out = model(
            drug_a_seq=drug_a,
            drug_b_seq=drug_b,
            drug_a_graph=None,
            drug_b_graph=None,
            cell_line=cell_line,
            classification_head_name="bclsmlpv2",
        )

    prob = torch.sigmoid(out).item()
    pred = 1 if prob >= 0.5 else 0
    preds.append(pred)
    labels.append(int(ex["label"]))

# Evaluation
acc = accuracy_score(labels, preds)
print(f"\n Gemini SMILES Accuracy: {acc:.4f}")
try:
    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"🧠 AUC: {auc:.4f}")
    print(f"🧠 F1 Score: {f1:.4f}")
except ValueError as e:
    print("⚠️ Could not compute AUC/F1:", e)

print("\n📊 Label Distribution:", Counter(labels))
print("📊 Prediction Distribution:", Counter(preds))
