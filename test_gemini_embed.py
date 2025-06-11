import torch
from omegaconf import OmegaConf
from dds.src.model.drug_gcn import GCNModel, GCNConfig
from dotenv import load_dotenv
load_dotenv()


# Mock args with Gemini enabled
args = OmegaConf.structured(GCNConfig(use_gemini=True))
model = GCNModel.build_model(args, task=None)

# Example input
drug_a = {"smiles": "CCO"}  # Ethanol
drug_b = {"smiles": "CC(=O)O"}  # Acetic acid
cell_line = torch.tensor([0], dtype=torch.long)

model.register_classification_head("bclsmlpv2", num_classes=1)

output = model(
    drug_a_seq=None,
    drug_b_seq=None,
    drug_a_graph=drug_a,
    drug_b_graph=drug_b,
    cell_line=cell_line,
    classification_head_name="bclsmlpv2",
)

print("Model output:", output)
