# from typing import Optional, Callable
# import torch
# import torch.nn.functional as F
# from torch.nn.modules.sparse import Embedding
# from torch_geometric.nn import MessagePassing
# from torch_scatter import scatter
# from torch import nn, Tensor
# from molecule.features import get_atom_feature_dims, get_bond_feature_dims
# from fairseq import utils
# from torch_geometric.nn import global_max_pool, global_mean_pool, global_sort_pool
# from torch_geometric.utils import add_self_loops, remove_self_loops
# from torch_geometric.utils.num_nodes import maybe_num_nodes

# from torch_geometric.nn.conv import GATv2Conv
# from torch_geometric.nn.inits import glorot, zeros
# from typing import Optional, Tuple, Union
# from torch_sparse import SparseTensor, set_diag
# from torch_geometric.typing import Adj, OptTensor, PairTensor
# from torch_scatter import gather_csr, segment_csr


# @torch.jit.script
# def CustomSoftmax(src: Tensor, index: Optional[Tensor] = None,
#             ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
#             dim: int = 0) -> Tensor:
    
#     src_32 = src.float()
#     if ptr is not None:
#         dim = dim + src_32.dim() if dim < 0 else dim
#         size = ([1] * dim) + [-1]
#         ptr = ptr.view(size)
#         src_max = gather_csr(segment_csr(src_32, ptr, reduce='max'), ptr)
#         out = (src_32 - src_max).exp()
#         out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
#     elif index is not None:
#         N = maybe_num_nodes(index, num_nodes)
#         src_max = scatter(src_32, index, dim, dim_size=N, reduce='max')
#         src_max = src_max.index_select(dim, index)
#         out = (src_32 - src_max).exp()
#         out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
#         out_sum = out_sum.index_select(dim, index)
#     else:
#         raise NotImplementedError

#     return (out / (out_sum + 1e-16))#.type_as(src)

# class CustomMessagePassing(MessagePassing):
#     def __init__(self, aggr: Optional[str] = "maxminmean", embed_dim: Optional[int] = None, node_dim: int = 0):
#         if aggr in ['maxminmean']:
#             super().__init__(aggr=None, node_dim=node_dim)
#             self.aggr = aggr
#             assert embed_dim is not None
#             self.aggrmlp = nn.Linear(3 * embed_dim, embed_dim)
#         else:
#             super().__init__(aggr=aggr)

#     def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor],
#                   dim_size: Optional[int]) -> Tensor:
#         if self.aggr in ['maxminmean']:
#             inputs_fp32 = inputs#.float()
#             input_max = scatter(inputs_fp32,
#                                 index,
#                                 dim=self.node_dim,
#                                 dim_size=dim_size,
#                                 reduce='max')
#             input_min = scatter(inputs_fp32,
#                                 index,
#                                 dim=self.node_dim,
#                                 dim_size=dim_size,
#                                 reduce='min')
#             input_mean = scatter(inputs_fp32,
#                                  index,
#                                  dim=self.node_dim,
#                                  dim_size=dim_size,
#                                  reduce='mean')
#             aggr_out = torch.cat([input_max, input_min, input_mean], dim=-1)#.half()#.type_as(inputs)
#             aggr_out = self.aggrmlp(aggr_out)
#             return aggr_out
#         else:
#             return super().aggregate(inputs, index, ptr, dim_size)


# class MulOnehotEncoder(nn.Module):
#     def __init__(self, embed_dim, get_feature_dims: Callable):
#         super().__init__()
#         self.atom_embedding_list = nn.ModuleList()

#         for dim in get_feature_dims():
#             emb = nn.Embedding(dim, embed_dim)
#             nn.init.xavier_uniform_(emb.weight.data)
#             self.atom_embedding_list.append(emb)

#     def forward(self, x):
#         x_embedding = 0
#         for i in range(x.shape[1]):
#             x_embedding = x_embedding + self.atom_embedding_list[i](x[:, i])
#         return x_embedding


# class GATv2ConvLayer(CustomMessagePassing):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  aggr,
#                  heads,
#                  negative_slope=0.2,
#                  dropout=0.0,
#                  add_self_loops=True,
#                  bias=True,
#                  share_weights=False,
#                  concat=True,
#                  fill_value='mean',
#                  edge_dim=None,
#                  **kwargs):
#         super().__init__(aggr, embed_dim=in_channels, node_dim=0)
#         # super().__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#         self.edge_dim = edge_dim
#         self.fill_value = fill_value
#         self.share_weights = share_weights

#         self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=bias)
#         if share_weights:
#             self.lin_r = self.lin_l
#         else:
#             self.lin_r = nn.Linear(in_channels, heads * out_channels,)

#         self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

#         if edge_dim is not None:
#             self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
#         else:
#             self.lin_edge = None

#         if bias and concat:
#             self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self._alpha = None

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_l.reset_parameters()
#         self.lin_r.reset_parameters()
#         if self.lin_edge is not None:
#             self.lin_edge.reset_parameters()
#         glorot(self.att)
#         zeros(self.bias)
    
#     def forward(self, x, edge_index, edge_attr=None):
#         H, C = self.heads, self.out_channels

#         x_l = self.lin_l(x).view(-1, H, C)
#         if self.share_weights:
#             x_r = x_l
#         else:
#             x_r = self.lin_r(x).view(-1, H, C)
        
#         assert x_l is not None
#         assert x_r is not None

#         if self.add_self_loops:
#             num_nodes = min(x_l.size(0), x_r.size(0))
#             edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
#             edge_index, edge_attr = add_self_loops(
#                 edge_index, edge_attr, fill_value=self.fill_value,
#                 num_nodes=num_nodes)
        
#         out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
#                             size=None)
        
#         alpha = self._alpha
#         self._alpha = None

#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         if self.bias is not None:
#             out += self.bias
        
#         return out
    
#     # def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
#     #             edge_attr: OptTensor = None,
#     #             return_attention_weights: bool = None):
#     #     # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
#     #     # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
#     #     # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
#     #     # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
#     #     r"""
#     #     Args:
#     #         return_attention_weights (bool, optional): If set to :obj:`True`,
#     #             will additionally return the tuple
#     #             :obj:`(edge_index, attention_weights)`, holding the computed
#     #             attention weights for each edge. (default: :obj:`None`)
#     #     """
#     #     H, C = self.heads, self.out_channels

#     #     x_l: OptTensor = None
#     #     x_r: OptTensor = None
#     #     if isinstance(x, Tensor):
#     #         assert x.dim() == 2
#     #         x_l = self.lin_l(x).view(-1, H, C)
#     #         if self.share_weights:
#     #             x_r = x_l
#     #         else:
#     #             x_r = self.lin_r(x).view(-1, H, C)
#     #     else:
#     #         x_l, x_r = x[0], x[1]
#     #         assert x[0].dim() == 2
#     #         x_l = self.lin_l(x_l).view(-1, H, C)
#     #         if x_r is not None:
#     #             x_r = self.lin_r(x_r).view(-1, H, C)

#     #     assert x_l is not None
#     #     assert x_r is not None

#     #     if self.add_self_loops:
#     #         if isinstance(edge_index, Tensor):
#     #             num_nodes = x_l.size(0)
#     #             if x_r is not None:
#     #                 num_nodes = min(num_nodes, x_r.size(0))
#     #             edge_index, edge_attr = remove_self_loops(
#     #                 edge_index, edge_attr)
#     #             edge_index, edge_attr = add_self_loops(
#     #                 edge_index, edge_attr, fill_value=self.fill_value,
#     #                 num_nodes=num_nodes)
#     #         elif isinstance(edge_index, SparseTensor):
#     #             if self.edge_dim is None:
#     #                 edge_index = set_diag(edge_index)
#     #             else:
#     #                 raise NotImplementedError(
#     #                     "The usage of 'edge_attr' and 'add_self_loops' "
#     #                     "simultaneously is currently not yet supported for "
#     #                     "'edge_index' in a 'SparseTensor' form")

#     #     # propagate_type: (x: PairTensor, edge_attr: OptTensor)
#     #     out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
#     #                          size=None)

#     #     alpha = self._alpha
#     #     self._alpha = None

#     #     if self.concat:
#     #         out = out.view(-1, self.heads * self.out_channels)
#     #     else:
#     #         out = out.mean(dim=1)

#     #     if self.bias is not None:
#     #         out += self.bias

#     #     if isinstance(return_attention_weights, bool):
#     #         assert alpha is not None
#     #         if isinstance(edge_index, Tensor):
#     #             return out, (edge_index, alpha)
#     #         elif isinstance(edge_index, SparseTensor):
#     #             return out, edge_index.set_value(alpha, layout='coo')
#     #     else:
#     #         return out
#     def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
#         x = x_i + x_j

#         if edge_attr is not None:
#             if edge_attr.dim() == 1:
#                 edge_attr = edge_attr.view(-1, 1)
#             assert self.lin_edge is not None
#             edge_attr = self.lin_edge(edge_attr)
#             edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
#             x += edge_attr
        
#         x = F.leaky_relu(x, self.negative_slope)
#         alpha = (x * self.att).sum(dim=-1)
#         alpha = CustomSoftmax(alpha, index, ptr, size_i)
#         self._alpha = alpha
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return x_j * alpha.unsqueeze(-1)


# def get_norm_layer(norm, fea_dim):
#     norm = norm.lower()

#     if norm == 'layer':
#         return nn.LayerNorm(fea_dim)
#     elif norm == "batch":
#         return nn.BatchNorm1d(fea_dim)
#     else:
#         raise NotImplementedError()

# class AtomHead(nn.Module):
#     def __init__(self, emb_dim, output_dim, activation_fn, weight=None, norm=None):
#         super().__init__()
#         self.dense = nn.Linear(emb_dim, emb_dim)
#         self.activation_fn = utils.get_activation_fn(activation_fn)
#         self.norm = get_norm_layer(norm, emb_dim)

#         if weight is None:
#             weight = nn.Linear(emb_dim, output_dim, bias=False).weight
#         self.weight = weight
#         self.bias = nn.Parameter(torch.zeros(output_dim))

#     def forward(self, node_features, masked_atom):
#         if masked_atom is not None:
#             node_features = node_features[masked_atom, :]

#         x = self.dense(node_features)
#         x = self.activation_fn(x)
#         x = self.norm(x)
#         x = F.linear(x, self.weight) + self.bias
#         return x

# class GATv2(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.num_layers = args.gnn_number_layer
#         self.dropout = args.gnn_dropout
#         self.conv_encode_edge = args.conv_encode_edge
#         self.embed_dim = args.gnn_embed_dim
#         self.aggr = args.gnn_aggr
#         self.norm = args.gnn_norm
#         self.heads = args.gat_heads

#         self.gcns = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         self.activation_fn = utils.get_activation_fn(getattr(args, 'gnn_activation_fn', 'relu'))


#         for _ in range(self.num_layers):
#             self.gcns.append(
#                 GATv2ConvLayer(
#                     self.embed_dim,
#                     self.embed_dim,
#                     self.aggr,
#                     heads=self.heads,
#                     concat=False,
#                     dropout=self.dropout,
#                     edge_dim=self.embed_dim,
#                 )
#             )
#             self.norms.append(get_norm_layer(self.norm, self.embed_dim))
        
#         self.atom_encoder = MulOnehotEncoder(self.embed_dim, get_atom_feature_dims)
#         if not self.conv_encode_edge:
#             self.bond_encoder = MulOnehotEncoder(self.embed_dim, get_bond_feature_dims)

#         self.graph_pred_linear = nn.Identity()
#         self.output_features = 2 * self.embed_dim
#         self.atom_head = AtomHead(self.embed_dim,
#                                   get_atom_feature_dims()[0],
#                                   getattr(args, 'gnn_activation_fn', 'relu'),
#                                   norm=self.norm,
#                                   weight=self.atom_encoder.atom_embedding_list[0].weight)

#     def forward(self, graph, masked_tokens=None, features_only=False):
#         x = graph.x
#         edge_index = graph.edge_index
#         edge_attr = graph.edge_attr
#         batch = graph.batch

#         h = self.atom_encoder(x)

#         if self.conv_encode_edge:
#             edge_emb = edge_attr
#         else:
#             edge_emb = self.bond_encoder(edge_attr)
        
#         h = self.gcns[0](h, edge_index, edge_emb)

#         for layer in range(1, self.num_layers):
#             residual = h
#             h = self.norms[layer](h)
#             h = self.activation_fn(h)
#             h = F.dropout(h, p=self.dropout, training=self.training)
#             h = self.gcns[layer](h, edge_index, edge_emb)
#             h = h + residual

#         h = self.norms[0](h)
#         h = self.activation_fn(h)
#         node_fea = F.dropout(h, p=self.dropout, training=self.training)

#         graph_fea = self.pool(node_fea, batch)
        
#         if not features_only:
#             atom_pred = self.atom_head(node_fea, masked_tokens)
#         else:
#             atom_pred = None

#         return (graph_fea, node_fea), atom_pred

    
#     def pool(self, h, batch):
#         h_fp32 = h.float()
#         h_max = global_max_pool(h_fp32, batch)
#         h_mean = global_mean_pool(h_fp32, batch)
#         h = torch.cat([h_max, h_mean], dim=-1).type_as(h)
#         h = self.graph_pred_linear(h)
#         return h
