import math
from optparse import Option
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    MultiheadAttention,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class TransformerCrossAttnLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.cross_attn = self.build_cross_attention(self.embed_dim, args)
        self.cross_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.cross_encoder_normalize_before
        self.fc_input = self.build_fc_input(
            self.embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc_output = self.build_fc_output(
            self.embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc_input(self, input_dim, output_dim, q_noise, qn_block_size):
        return apply_quant_noise_(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc_output(self, input_dim, output_dim, q_noise, qn_block_size):
        return apply_quant_noise_(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_cross_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x
    
    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask: Optional[Tuple], attn_mask: Optional[Tuple] = None):
        """
        Args:
            x (Tuple): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (Tuple): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (Tuple): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        src_x, tgt_x = x[0], x[1]
        if attn_mask is not None:
            src_attn_mask, tgt_attn_mask = attn_mask[0], attn_mask[1]
            src_attn_mask = src_attn_mask.masked_fill(src_attn_mask.to(torch.bool), -1e8)
            tgt_attn_mask = tgt_attn_mask.masked_fill(tgt_attn_mask.to(torch.bool), -1e8)
        
        if encoder_padding_mask is not None:
            src_encoder_padding_mask, tgt_encoder_padding_mask = encoder_padding_mask[0], encoder_padding_mask[1]
        
        src_cls, tgt_cls = src_x[-1:, :, :], tgt_x[-1:, :, :]
        src_cls, tgt_cls = self.fc_input(src_cls), self.fc_input(tgt_cls)

        src_feats, tgt_feats = src_x[:-1, :, :], tgt_x[:-1, :, :]
        src_residual, tgt_residual = src_cls, tgt_cls

        if self.normalize_before:
            src_x = self.cross_attn_layer_norm(src_x)
            tgt_x = self.cross_attn_layer_norm(tgt_x)
        
        src_cls, _ = self.cross_attn(
            query=src_cls,
            key=torch.cat([tgt_feats, src_cls], dim=0),
            value=torch.cat([tgt_feats, src_cls], dim=0),
            key_padding_mask=tgt_encoder_padding_mask,
        ) # to check static_kv

        src_cls = self.dropout_module(src_cls)
        src_cls = self.residual_connection(src_cls, src_residual)
        

        tgt_cls, _ = self.cross_attn(
            query=tgt_cls,
            key=torch.cat([src_feats, tgt_cls], dim=0),
            value=torch.cat([src_feats, tgt_cls], dim=0),
            key_padding_mask=src_encoder_padding_mask,
        ) # to check static_kv

        tgt_cls = self.dropout_module(tgt_cls)
        tgt_cls = self.residual_connection(tgt_cls, tgt_residual)


        if not self.normalize_before:
            src_x = self.cross_attn_layer_norm(src_x)
            tgt_x = self.cross_attn_layer_norm(tgt_x)

        src_cls = self.fc_output(src_cls)
        tgt_cls = self.fc_output(tgt_cls)

        src_x = torch.cat([src_feats, src_cls], dim=0)
        tgt_x = torch.cat([tgt_feats, tgt_cls], dim=0)

        return src_x, tgt_x

        


class TransformerPairEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
            self.cross_layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
            self.cross_layers = nn.ModuleList([])
        
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )

        self.cross_layers.extend(
            [self.build_cross_attn_layer(args) for i in range(args.cross_encoder_layers)]
        )
        
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def build_cross_attn_layer(self, args):
        layer = TransformerCrossAttnLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    
    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        tgt_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        tgt_lenghts: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(src_tokens,
                                       tgt_tokens,
                                       src_lengths,
                                       tgt_lenghts,
                                       return_all_hiddens,
                                       token_embeddings)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        tgt_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        tgt_lenghts: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        src_encoder_padding_mask = src_tokens.eq(self.padding_idx)
        src_has_pads = (src_tokens.device.type == "xla" or src_encoder_padding_mask.any())
        tgt_encoder_padding_mask = tgt_tokens.eq(self.padding_idx)
        tgt_has_pads = (tgt_tokens.device.type == "xla" or tgt_encoder_padding_mask.any())

        src_x, src_encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        tgt_x, tgt_encoder_embedding = self.forward_embedding(tgt_tokens, token_embeddings)

        # account for padding while computing the representation
        if src_has_pads:
            src_x = src_x * (1 - src_encoder_padding_mask.unsqueeze(-1).type_as(src_x))
        if tgt_has_pads:
            tgt_x = tgt_x * (1 - tgt_encoder_padding_mask.unsqueeze(-1).type_as(tgt_x))
        
        # B x T x C -> T x B x C
        src_x = src_x.transpose(0, 1)
        tgt_x = tgt_x.transpose(0, 1)

        src_encoder_states = []
        tgt_encoder_states = []

        if return_all_hiddens:
            src_encoder_states.append(src_x)
            tgt_encoder_states.append(tgt_x)
        
        # encoder layers
        for layer, cross_layer in zip(self.layers, self.cross_layers):
            # self attention
            src_x = layer(
                src_x, encoder_padding_mask=src_encoder_padding_mask if src_has_pads else None
            )
            tgt_x = layer(
                tgt_x, encoder_padding_mask=tgt_encoder_padding_mask if tgt_has_pads else None
            )

            # cross attention
            src_x, tgt_x = cross_layer(
                (src_x, tgt_x), 
                (src_encoder_padding_mask, tgt_encoder_padding_mask)
                )

            if return_all_hiddens:
                assert src_encoder_states is not None and tgt_encoder_states is not None
                src_encoder_states.append(src_x)
                tgt_encoder_states.append(tgt_x)
        
        if self.layer_norm is not None:
            src_x = self.layer_norm(src_x)
            tgt_x = self.layer_norm(tgt_x)
        
        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [src_x, tgt_x],  # T x B x C
            "encoder_padding_mask": [src_encoder_padding_mask, tgt_encoder_padding_mask],  # B x T
            "encoder_embedding": [src_encoder_embedding, tgt_encoder_embedding],  # B x T x C
            "encoder_states": [src_encoder_states, tgt_encoder_states],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }