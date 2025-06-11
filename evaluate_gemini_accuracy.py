import logging
logging.getLogger().setLevel(logging.ERROR)

from tqdm import tqdm
import torch
from omegaconf import OmegaConf
from dds.src.model.drug_gemini import GeminiModel, GeminiConfig
from gemini_test_loader import load_test_data
from load_cell_features import load_cell_tpm_features
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
from collections import Counter

load_dotenv()

args = OmegaConf.structured(GeminiConfig(
    gnn_embed_dim=768,
    embedding_source="image_text"
))
model = GeminiModel(args)
model.register_classification_head("bclsmlpv2", num_classes=1)

model.load_state_dict(torch.load("best_model_epoch15.pt"))  
model.eval()

examples = load_test_data()
tpm_df = load_cell_tpm_features()

preds, probs, labels = [], [], []

for ex in tqdm(examples, desc="Evaluating Gemini Image-Text Model"):
    try:
        drug_a = {"smiles": ex["drug_a_name"].strip()}
        drug_b = {"smiles": ex["drug_b_name"].strip()}
        cell_line = torch.tensor(ex["cell_idx"], dtype=torch.long)

        with torch.no_grad():
            output = model(
                drug_a_seq=drug_a,
                drug_b_seq=drug_b,
                drug_a_graph=None,
                drug_b_graph=None,
                cell_line=cell_line,
                classification_head_name="bclsmlpv2"
            )

        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= 0.5 else 0

        preds.append(pred)
        probs.append(prob)
        labels.append(int(ex["label"]))
    except Exception as e:
        continue

acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)
auc = roc_auc_score(labels, probs)

print(f"\n Gemini+ImageText Accuracy: {acc:.4f}")
print(f" F1 Score: {f1:.4f}")
print(f" AUC Score: {auc:.4f}")

print("\n Label Distribution:", Counter(labels))
print(" Prediction Distribution:", Counter(preds))
print("\n Classification Report:\n", classification_report(labels, preds, digits=4))
print(" Confusion Matrix:\n", confusion_matrix(labels, preds))
