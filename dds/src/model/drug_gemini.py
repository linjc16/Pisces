import logging
import os
import pandas as pd
import torch
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from .heads import BinaryClassMLPv2Head
from .heads_ppi import BinaryClassMLPPPIv2Head

logger = logging.getLogger(__name__)

@dataclass
class GeminiConfig(FairseqDataclass):
    gnn_embed_dim: int = field(default=768)
    classification_head_name: str = field(default='')
    pooler_activation_fn: str = field(default='tanh')
    pooler_dropout: float = field(default=0.1)
    n_memory: int = field(default=32)
    embedding_source: str = field(default="smiles")  

@register_model("drug_gemini", dataclass=GeminiConfig)
class GeminiModel(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.classification_heads = torch.nn.ModuleDict()
        self.embedding_dim = 768
        self.gemini_cache = self.load_gemini_embedding_cache()

    def load_gemini_embedding_cache(self):
        if self.args.embedding_source == "smiles":
            filename = "data/gemini_smiles_embeddings.csv"
        elif self.args.embedding_source == "text":
            filename = "data/gemini_text_embeddings.csv"
        elif self.args.embedding_source == "image_text":
            filename = "data/gemini_image_text_embeddings.csv"
        else:
            raise ValueError(f"❌ Unknown embedding_source: {self.args.embedding_source}")

        if not os.path.exists(filename):
            raise FileNotFoundError(f"❌ Expected embedding file not found: {filename}")

        df = pd.read_csv(filename)
        embedding_dict = {}
        for _, row in df.iterrows():
            emb = [float(x) for x in row["embedding"].split(",")]
            if len(emb) != 768:
                logger.warning(f"⚠️ Skipping {row['drug_name']} due to invalid length {len(emb)}")
                continue
            embedding_dict[row["drug_name"]] = torch.tensor(emb).unsqueeze(0)
        return embedding_dict

    def get_embedding(self, drug_name: str):
        if drug_name not in self.gemini_cache:
            raise ValueError(f"Drug '{drug_name}' not found in Gemini cache.")
        return self.gemini_cache[drug_name]

    def forward(self, drug_a_seq, drug_b_seq, drug_a_graph, drug_b_graph, cell_line,
                classification_head_name=None, **kwargs):
        name_a = drug_a_seq["smiles"]  
        name_b = drug_b_seq["smiles"]
        enc_a = self.get_embedding(name_a)
        enc_b = self.get_embedding(name_b)
        return self.classification_heads[classification_head_name](enc_a, enc_b, cell_line)

    def forward_embed(self, drug_a_seq, drug_b_seq, **kwargs):
        name_a = drug_a_seq["smiles"]
        name_b = drug_b_seq["smiles"]
        return self.get_embedding(name_a), self.get_embedding(name_b)

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out.out_features
            prev_inner_dim = self.classification_heads[name].fc2.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    f'Re-registering head "{name}" with num_classes={num_classes} (prev={prev_num_classes}) '
                    f'and inner_dim={inner_dim} (prev={prev_inner_dim})'
                )

        if name == 'bclsmlpv2':
            self.classification_heads[name] = BinaryClassMLPv2Head(
                input_dim=self.embedding_dim,
                inner_dim=inner_dim or self.embedding_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        elif name == 'bclsmlpppiv2':
            self.classification_heads[name] = BinaryClassMLPPPIv2Head(
                input_dim=self.embedding_dim,
                inner_dim=inner_dim or self.embedding_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
                n_memory=self.args.n_memory,
            )
        else:
            raise NotImplementedError(f"Classification head '{name}' not implemented.")

@register_model_architecture("drug_gemini", "drug_gemini_base")
def base_architecture(args):
    args.gnn_embed_dim = getattr(args, "gnn_embed_dim", 768)
