import logging
from dataclasses import dataclass, field
from typing import Optional, overload
from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (FairseqEncoder, BaseFairseqModel, register_model, register_model_architecture)
import torch
import torch.nn.functional as F
from torch import nn
from fairseq.models.roberta import RobertaLMHead
from fairseq.models.transformer_pair_seq import TransformerPairEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from omegaconf import II
from .heads import BinaryClassMLPHead, BinaryClassMLPv2Head, BinaryClassMLPv2NonormHead
from .heads_ppi import BinaryClassMLPPPIHead, BinaryClassMLPPPIv2Head, BinaryClassMLPPPIInnerMixHead, \
        BinaryClassMLPPPIInterMixHead, BinaryClassMLPAttnPPIHead
from .heads_prot_seq import BinaryClassProtSeqFrozenMLPHead
from .heads_pair import BinaryClassMLPPairHead, BinaryClassMLPPPIv2PairHead
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
    cross_encoder_normalize_before: bool = field(default=False)
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


@register_model("drug_pair_transfomer_seq", dataclass=DrugTransformerConfig)
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
        
        drug_pair_seq = drug_a_seq
        drug_pair_seq['tgt_tokens'] = drug_b_seq['src_tokens']
        drug_pair_seq['tgt_lengths'] = drug_b_seq['src_lengths']

        enc_a, enc_b, _ = self.encoder(**drug_pair_seq, features_only=features_only, **kwargs)
        
        enc_a = self.get_cls(enc_a)
        enc_b = self.get_cls(enc_b)

        x = self.classification_heads[classification_head_name](enc_a, enc_b, cell_line)
    
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

        if name == 'bclsmlp':
            
            self.classification_heads[name] = BinaryClassMLPHead(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        
        elif name == 'bclsmlpppi':
            self.classification_heads[name] = BinaryClassMLPPPIHead(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        elif name == 'bclsmlpv2':
            self.classification_heads[name] = BinaryClassMLPv2Head(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        elif name == 'bclsmlppair':
            self.classification_heads[name] = BinaryClassMLPPairHead(
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
        elif name == 'bclsmlpv2nonorm':
            self.classification_heads[name] = BinaryClassMLPv2NonormHead(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        elif name == 'bclsmlpppiv2pair':
            self.classification_heads[name] = BinaryClassMLPPPIv2PairHead(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
                n_memory=self.args.n_memory,
            )
        elif name == 'bclsmlpattnppi':
            self.classification_heads[name] = BinaryClassMLPAttnPPIHead(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
                n_memory=self.args.n_memory,
            )
        elif name == 'bclsmlpppiInnermix':
            self.classification_heads[name] = BinaryClassMLPPPIInnerMixHead(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        elif name == 'bclsmlpppiIntermix':
            self.classification_heads[name] = BinaryClassMLPPPIInterMixHead(
                input_dim=getattr(self.encoder, "output_features", self.args.encoder_embed_dim),
                inner_dim=inner_dim or self.args.encoder_embed_dim,
                num_classes=num_classes,
                actionvation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
            )
        
        else:
            raise NotImplementedError('No Implemented by DDI')

    def max_positions(self):
        return self.args.max_positions

class TrEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        # set any missing default values
        base_architecture(args)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)

        self.lm_head = self.build_lm_head(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            ),
        )
    
    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerPairEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def forward(
        self,
        src_tokens,
        tgt_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        src_features, tgt_features, extra = self.extract_features(src_tokens, tgt_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            src_features = self.output_layer(src_features, masked_tokens=masked_tokens)
            tgt_features = self.output_layer(tgt_features, masked_tokens=masked_tokens)
        else:
            x = None
        
        return src_features, tgt_features, x

    def extract_features(self, src_tokens, tgt_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            tgt_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        src_features = encoder_out["encoder_out"][0].transpose(0, 1)
        tgt_features = encoder_out["encoder_out"][1].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return src_features, tgt_features, {"inner_states": inner_states}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


# @register_model_architecture("drug_pair_transfomer", "drug_pair_transfomer_tiny")
# def tiny_architecture(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
#     args.encoder_layers = getattr(args, "encoder_layers", 2)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
#     return base_architecture(args)

@register_model_architecture("drug_pair_transfomer_seq", "drug_pair_transfomer_seq_tiny")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.cross_encoder_layers = getattr(args, "cross_encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

@register_model_architecture("drug_pair_transfomer_seq", "drug_pair_transfomer_seq_base")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.cross_encoder_layers = getattr(args, "cross_encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

@register_model_architecture("drug_pair_transfomer_seq", "drug_pair_transfomer_seq_large")
def large_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
