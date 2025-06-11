import os
import torch
import pandas as pd
from omegaconf import OmegaConf
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import WeightedRandomSampler
from dds.src.model.drug_gemini import GeminiModel, GeminiConfig
from gemini_test_loader import load_test_data
from load_cell_features import load_cell_tpm_features

# 🔧 Setup
load_dotenv()
examples = load_test_data()
tpm_df = load_cell_tpm_features()

# 📦 Model initialization
args = OmegaConf.structured(GeminiConfig(
    gnn_embed_dim=768,
    embedding_source="image_text"
))
model = GeminiModel(args)
model.register_classification_head("bclsmlpv2", num_classes=1)
model.train()

# 🧠 Calculate class imbalance
labels = pd.Series([ex["label"] for ex in examples])
label_counts = labels.value_counts()
neg, pos = label_counts[0], label_counts[1]
pos_weight = torch.tensor([neg / pos])
print(f"Using pos_weight={pos_weight.item():.2f}")

# 🚀 Optimizer, Loss, Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

# 📚 Prepare training data
inputs, targets = [], []
for ex in shuffle(examples):
    drug_a_name = ex["drug_a_name"].strip()
    drug_b_name = ex["drug_b_name"].strip()
    if drug_a_name not in model.gemini_cache or drug_b_name not in model.gemini_cache:
        continue

    drug_a = {"smiles": drug_a_name}
    drug_b = {"smiles": drug_b_name}
    cell_line = torch.tensor([ex["cell_idx"]], dtype=torch.long)
    label = torch.tensor([ex["label"]], dtype=torch.float32)

    inputs.append((drug_a, drug_b, cell_line))
    targets.append(label)

print(f"✅ Total training samples loaded: {len(inputs)}")

# 🧮 For optional weighted sampling
label_tensor = torch.tensor([int(t.item()) for t in targets])
class_weights = 1.0 / torch.bincount(label_tensor)
sample_weights = class_weights[label_tensor]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# 🔁 Training loop
best_f1 = 0.0

for epoch in range(15):
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for idx in tqdm(list(sampler), desc=f"Epoch {epoch + 1}"):
        drug_a, drug_b, cell_line = inputs[idx]
        label = targets[idx]

        model.zero_grad()
        out = model(
            drug_a_seq=drug_a,
            drug_b_seq=drug_b,
            drug_a_graph=None,
            drug_b_graph=None,
            cell_line=cell_line,
            classification_head_name="bclsmlpv2"
        )

        loss = criterion(out.view(-1), label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

        prob = torch.sigmoid(out).item()
        pred = 1 if prob >= 0.5 else 0
        all_probs.append(prob)
        all_preds.append(pred)
        all_labels.append(label.item())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    scheduler.step()

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.pt")

    print(f"\n✅ Epoch {epoch + 1} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
