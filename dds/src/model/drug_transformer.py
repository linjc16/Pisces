import logging
from dataclasses import dataclass, field
from typing import Optional
from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (BaseFairseqModel, register_model, register_model_architecture)
import torch
import torch.nn.functional as F
from torch import nn
from fairseq.models.roberta import RobertaEncoder
from omegaconf import II
from .heads import BinaryClassMLPv2Head
from .heads_ppi import BinaryClassMLPPPIv2Head
from .trash.heads_prot_seq import BinaryClassProtSeqFrozenMLPHead

import pdb

logger = logging.getLogger(__name__)


@dataclass
class DrugTransformerConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(default='gelu', )
    dropout: float = field(default=0.1)
    attention_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.0)
    relu_dropout: float = field(default=0.0)
    encoder_embed_path: Optional[str] = field(default=None)
    encoder_embed_dim: int = field(default=768)
    encoder_ffn_embed_dim: int = field(default=3072)
    encoder_layers: int = field(default=12)
    cross_encoder_layers: int = field(default=6)
    encoder_attention_heads: int = field(default=12)
    encoder_normalize_before: bool = field(default=False)
    encoder_learned_pos: bool = field(default=True)
    layernorm_embedding: bool = field(default=True)
    no_scale_embedding: bool = field(default=True)
    max_positions: int = field(default=512)

    classification_head_name: str = field(default='')
    load_checkpoint_heads: bool = field(default=False)


    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={"help": "scalar quantization noise and scalar quantization at training time"},
    )
    # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
    spectral_norm_classification_head: bool = field(default=False)

    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    encoder_layerdrop: float = field(default=0.0,
                                     metadata={"help": "LayerDrop probability for decoder"})
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={"help": "which layers to *keep* when pruning as a comma-separated list"},
    )
    max_source_positions: int = II("model.max_positions")
    no_token_positional_embeddings: bool = field(default=False)
    pooler_activation_fn: str = field(default='tanh')
    pooler_dropout: float = field(default=0.0)
    untie_weights_roberta: bool = field(default=False)
    adaptive_input: bool = field(default=False)
    n_memory: int = field(default=32)
    
    skip_update_state_dict: bool = field(default=False,
        metadata={"help": "Don't update state dict when load pretrained model weight"},
    )


@register_model("drug_transfomer", dataclass=DrugTransformerConfig)
class DrugTransfomerModel(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, args, task):
        
        base_architecture(args)
        encoder = TrEncoder(args, task.source_dictionary)
        
        return cls(args, encoder)

    def forward(self,
                drug_a_seq,
                drug_b_seq,
                drug_a_graph,
                drug_b_graph,
                cell_line,
                features_only=False,
                classification_head_name=None,
                **kwargs):
        
        if classification_head_name is not None:
            features_only = True
        
        enc_a, _ = self.encoder(**drug_a_seq, features_only=features_only, **kwargs)
        enc_b, _ = self.encoder(**drug_b_seq, features_only=features_only, **kwargs)
        
        enc_a = self.get_cls(enc_a)
        enc_b = self.get_cls(enc_b)

        x = self.classification_heads[classification_head_name](enc_a, enc_b, cell_line)
    
        return x

    def forward_embed(self,
                drug_a_seq,
                drug_b_seq,
                drug_a_graph,
                drug_b_graph,
                cell_line,
                features_only=False,
                classification_head_name=None,
                **kwargs):
        
        if classification_head_name is not None:
            features_only = True
        
        enc_a, _ = self.encoder(**drug_a_graph, features_only=features_only, **kwargs)
        enc_b, _ = self.encoder(**drug_b_graph, features_only=features_only, **kwargs)
        
        enc_a = self.get_cls(enc_a)
        enc_b = self.get_cls(enc_b)

        return enc_a, enc_b
    
    def forward_inter_mix(self,
                drug_a_seq,
                drug_b_seq,
                drug_a_graph,
                drug_b_graph,
                cell_line,
                targets,
                features_only=False,
                classification_head_name=None,
                **kwargs):
        
        if classification_head_name is not None:
            features_only = True
        
        enc_a, _ = self.encoder(**drug_a_seq, features_only=features_only, **kwargs)
        enc_b, _ = self.encoder(**drug_b_seq, features_only=features_only, **kwargs)
        
        enc_a = self.get_cls(enc_a)
        enc_b = self.get_cls(enc_b)

        x, labels = self.classification_heads[classification_head_name](enc_a, enc_b, cell_line, targets)
    
        return x, labels

    def forward_inter_mix_eval(self,
                drug_a_seq,
                drug_b_seq,
                drug_a_graph,
                drug_b_graph,
                cell_line,
                targets,
                features_only=False,
                classification_head_name=None,
                **kwargs):
        
        if classification_head_name is not None:
            features_only = True
        
        enc_a, _ = self.encoder(**drug_a_seq, features_only=features_only, **kwargs)
        enc_b, _ = self.encoder(**drug_b_seq, features_only=features_only, **kwargs)
        
        enc_a = self.get_cls(enc_a)
        enc_b = self.get_cls(enc_b)

        x = self.classification_heads[classification_head_name].forward_eval(enc_a, enc_b, cell_line)
    
        return x
    def get_cls(self, x):
        if x is None:
            return 0
        if isinstance(x, torch.Tensor):
            return x[:, -1, :]
        elif isinstance(x, tuple):
            return x[0]
        else:
            raise ValueError()

    def get_targets(self, target, input):
        return target

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning('re-registering head "{}" with num_classes {} (prev: {}) '
                               "and inner_dim {} (prev: {})".format(name, num_classes,
                                                                    prev_num_classes, inner_dim,
                                                                    prev_inner_dim))

        if name == 'bclsmlpv2':
            self.classification_heads[name] = BinaryClassMLPv2Head(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        elif name == 'bclsProtSeqFrozenmlp':
            self.classification_heads[name] = BinaryClassProtSeqFrozenMLPHead(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        
        elif name == 'bclsmlpppiv2':
            self.classification_heads[name] = BinaryClassMLPPPIv2Head(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
                n_memory=self.args.n_memory,
            )

        else:
            raise NotImplementedError('No Implemented by DDS')

    def max_positions(self):
        return self.args.max_positions

class TrEncoder(RobertaEncoder):
    def __init(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        features, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            x = self.output_layer(features, masked_tokens=masked_tokens)
        else:
            x = None
        
        return features, x


@register_model_architecture("drug_transfomer", "drug_transfomer_tiny")
def tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    return base_architecture(args)

@register_model_architecture("drug_transfomer", "drug_transfomer_base")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

@register_model_architecture("drug_transfomer", "drug_transfomer_large")
def large_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
