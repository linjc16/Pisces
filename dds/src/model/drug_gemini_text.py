import os
import torch
import pandas as pd
import logging
from dataclasses import dataclass, field
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.dataclass import FairseqDataclass
from .heads import BinaryClassMLPv2Head
from .heads_ppi import BinaryClassMLPPPIv2Head

logger = logging.getLogger(__name__)

@dataclass
class GeminiTextConfig(FairseqDataclass):
    embed_dim: int = field(default=512)
    classification_head_name: str = field(default='')
    pooler_activation_fn: str = field(default='tanh')
    pooler_dropout: float = field(default=0.1)
    n_memory: int = field(default=32)

@register_model("drug_gemini_text", dataclass=GeminiTextConfig)
class GeminiTextModel(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.classification_heads = torch.nn.ModuleDict()
        self.embedding_cache = self.load_embedding_cache()

    def load_embedding_cache(self):
        path = "data/gemini_text_embeddings.csv"
        if not os.path.exists(path):
            raise FileNotFoundError("❌ Text embeddings file not found. Run precompute_gemini_text_embeddings.py first.")
        df = pd.read_csv(path)
        cache = {}
        for _, row in df.iterrows():
            emb = torch.tensor([float(x) for x in row["embedding"].split(",")]).unsqueeze(0)
            cache[row["drug_name"]] = emb
        return cache

    def get_embedding(self, smiles: str):
        if smiles not in self.embedding_cache:
            raise ValueError(f"Drug '{smiles}' not found in Gemini text embedding cache.")
        return self.embedding_cache[smiles]

    def forward(self, drug_a_seq, drug_b_seq, drug_a_graph, drug_b_graph, cell_line,
                classification_head_name=None, **kwargs):
        emb_a = self.get_embedding(drug_a_seq["smiles"])
        emb_b = self.get_embedding(drug_b_seq["smiles"])
        return self.classification_heads[classification_head_name](emb_a, emb_b, cell_line)

    def forward_embed(self, drug_a_seq, drug_b_seq, **kwargs):
        return self.get_embedding(drug_a_seq["smiles"]), self.get_embedding(drug_b_seq["smiles"])

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        if name in self.classification_heads:
            logger.warning(f"Re-registering classification head '{name}'")
        if name == 'bclsmlpv2':
            self.classification_heads[name] = BinaryClassMLPv2Head(
                input_dim=self.args.embed_dim,
                inner_dim=inner_dim or self.args.embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        elif name == 'bclsmlpppiv2':
            self.classification_heads[name] = BinaryClassMLPPPIv2Head(
                input_dim=self.args.embed_dim,
                inner_dim=inner_dim or self.args.embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
                n_memory=self.args.n_memory,
            )
        else:
            raise NotImplementedError(f"Head '{name}' not implemented.")

@register_model_architecture("drug_gemini_text", "drug_gemini_text_base")
def base_architecture(args):
    args.embed_dim = getattr(args, "embed_dim", 512)
