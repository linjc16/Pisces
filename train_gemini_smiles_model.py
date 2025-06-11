import os
import torch
import pandas as pd
from omegaconf import OmegaConf
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dds.src.model.drug_gemini import GeminiModel, GeminiConfig
from gemini_test_loader import load_test_data
from load_cell_features import load_cell_tpm_features

# 🔧 Setup
load_dotenv()
examples = load_test_data()
tpm_df = load_cell_tpm_features()

# 📦 Model initialization
args = OmegaConf.structured(GeminiConfig(gnn_embed_dim=768))  # ✅ match Gemini embedding dim
model = GeminiModel(args)
model.register_classification_head("bclsmlpv2", num_classes=1)
model.train()

# 🧠 Calculate class imbalance for pos_weight
label_counts = pd.Series([ex["label"] for ex in examples]).value_counts()
neg, pos = label_counts[0], label_counts[1]
pos_weight = torch.tensor([neg / pos])  # e.g., ~18.19
print(f"Using pos_weight={pos_weight.item():.2f}")

# 🚀 Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 🔽 lowered LR
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 📚 Prepare training data
inputs = []
for ex in shuffle(examples):  # optional shuffle
    try:
        drug_a = {"smiles": ex["smiles_a"]}
        drug_b = {"smiles": ex["smiles_b"]}
        cell_line = torch.tensor([ex["cell_idx"]], dtype=torch.long)
        label = torch.tensor([ex["label"]], dtype=torch.float32)
        inputs.append((drug_a, drug_b, cell_line, label))
    except:
        continue

# 🔁 Training loop
for epoch in range(15):  # 🔄 train longer
    total_loss = 0.0
    all_preds, all_labels = [], []

    for drug_a, drug_b, cell_line, label in tqdm(inputs, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        out = model(
            drug_a_seq=drug_a,
            drug_b_seq=drug_b,
            drug_a_graph=None,
            drug_b_graph=None,
            cell_line=cell_line,
            classification_head_name="bclsmlpv2",
        )
        loss = criterion(out.view(-1), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        prob = torch.sigmoid(out).item()
        pred = 1 if prob >= 0.5 else 0
        all_preds.append(pred)
        all_labels.append(label.item())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0

    print(f"\n✅ Epoch {epoch + 1} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
