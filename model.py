import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import process
import scipy.sparse as sp

import dgl
import torch
import torch.nn as nn

from layer import Attentive, GCNConv_dense, GCNConv_dgl, GCN
from utils import *
# import process


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        # assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, out_channels)]
        # for _ in range(1, k-1):
        #     self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        # self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder

        self.lin = torch.nn.Linear(num_hidden, num_hidden)

    def forward(self, x_1: torch.Tensor,  x_2: torch. Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_1 = self.encoder(x_1, edge_index)
        h_2 = self.encoder(x_2, edge_index)

        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1)).unsqueeze(0)
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1)).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

    def embed(self, x, edge_index, sp_adj):
        h_1 = self.encoder(x, edge_index)

        h_2 = h_1.clone().squeeze(0)
        for i in range(5):
            h_2 = sp_adj @ h_2

        h_2 = h_2.unsqueeze(0)

        return h_1.detach(), h_2.detach()

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def loss_func(feat, cluster_centers):
    alpha = 1.0
    q = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    p = p.detach()

    log_q = torch.log(q)
    loss = F.kl_div(log_q, p)
    return loss, p, q

def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph

def top_k(raw_graph, K):
    _, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    del mask
    del raw_graph
    return sparse_graph

def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')

# def normalize_adj(learned_adj):
#     """Symmetrically normalize adjacency matrix.
#     Args:
#         adj: A torch.cuda.sparse.FloatTensor representing the adjacency matrix.
#     Returns:
#         A torch.cuda.sparse.FloatTensor representing the normalized adjacency matrix.
#     """
#     rowsum = learned_adj.to_dense().sum(dim=1)
#     d_inv_sqrt = torch.where(rowsum > 0, torch.pow(rowsum, -0.5), torch.zeros_like(rowsum))
#     d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()
#     return learned_adj.to_dense().mm(d_mat_inv_sqrt.to_dense()).t().mm(d_mat_inv_sqrt.to_dense()).to_sparse()

def normalize_adj(learned_adj):
    rowsum = learned_adj.sum(dim=1)
    d_inv_sqrt = torch.where(rowsum > 0, torch.pow(rowsum, -0.5), torch.zeros_like(rowsum))
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_adj = learned_adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)
    del d_mat_inv_sqrt
    del learned_adj
    return normalized_adj

def normalize_adj_lp(learned_adj):
    rowsum = learned_adj.sum(dim=1)
    d_inv_sqrt = torch.where(rowsum > 0, rowsum ** -0.5, rowsum)
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    normalized_adj = d_mat_inv_sqrt @ learned_adj @ d_mat_inv_sqrt
    del d_mat_inv_sqrt
    del learned_adj
    return normalized_adj
def graph_learner(features, k, non_linearity, i):
    """
    :param features: 向量
    :param k:
    :param non_linearity:
    :param i:
    :return:
    """
    embeddings = F.normalize(features, dim=1, p=2)
    #similarities0 = torch.mm(embeddings, embeddings.t())
    #sim = similarities0 - torch.diag_embed(torch.diag(similarities0))
    similarities = top_k(torch.mm(embeddings, embeddings.t()), k + 1)

    del embeddings
    del features
    torch.cuda.empty_cache()
    similarities = apply_non_linearity(similarities, non_linearity, i)
    learned_adj = process.symmetrize(similarities)
    del similarities
    # learned_adj = learned_adj.to_sparse()

    adj = normalize_adj(learned_adj)
    # adj2 = normalize_adj_lp(learned_adj)
    del learned_adj
    #learned_adj = adj
    torch.cuda.empty_cache()
    return adj

# def graph_learner( features, k, non_linearity, i):
#     """
#     :param features: 向量
#     :param k:
#     :param non_linearity:
#     :param i:
#     :return:
#     """
#
#     embeddings = F.normalize(features, dim=1, p=2)
#     similarities = torch.mm(embeddings, embeddings.t())
#     #similarities = torch.clamp(similarities, 0, 1)  # 每个值限定在0-1之间，一种截断
#     sim = similarities - torch.diag_embed(torch.diag(similarities))
#
#     similarities = top_k(similarities, k + 1)
#     similarities = apply_non_linearity(similarities, non_linearity, i)
#     learned_adj = process.symmetrize(similarities)  # 对称化
#     # learned_adj = normalize(sp.csr_matrix(learned_adj), 'sym', args.sparse)
#     learned_adj = sp.csr_matrix(learned_adj.cpu().numpy())
#     adj = process.normalize_adj(learned_adj)
#     learned_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
#
#     return learned_adj, similarities


def graph_learner2( features, k, non_linearity, i):
    """
    取消结构学习器，直接用学习到的向量更新adj
    :param features: 向量
    :param k:
    :param non_linearity:
    :param i:
    :return:

    """
    embeddings = F.normalize(features, dim=1, p=2)
    similarities = cal_similarity_graph(embeddings)
    similarities = top_k(similarities, k + 1)
    similarities = apply_non_linearity(similarities, non_linearity, i)
    return similarities



class FGP_learner(nn.Module):
    def __init__(self, features, k, knn_metric, i, sparse):
        super(FGP_learner, self).__init__()

        self.k = k
        self.knn_metric = knn_metric
        self.i = i
        self.sparse = sparse

        self.Adj = nn.Parameter(
            torch.from_numpy(process.nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i)))

    def forward(self, h):
        if not self.sparse:
            Adj = F.elu(self.Adj) + 1
        else:
            Adj = self.Adj.coalesce()
            Adj.values = F.elu(Adj.values()) + 1
        return Adj


class ATT_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act):
        super(ATT_learner, self).__init__()

        self.i = i
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentive(isize))
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        #weights = self.layers.modules().weight
        weights = self.layers



        return h,weights

    def forward(self, features):
        if self.sparse:
            embeddings,weights = self.internal_forward(features)
            rows, cols, values = process.knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj, weights
        else:
            embeddings,weights = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities,weights


class MLP_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, act):
        super(MLP_learner, self).__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, isize))
        else:
            self.layers.append(nn.Linear(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(isize, isize))
            self.layers.append(nn.Linear(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.act = act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = process.knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities


class GNN_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act, adj):
        super(GNN_learner, self).__init__()

        self.adj = adj
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GCNConv_dgl(isize, isize))
        else:
            self.layers.append(GCNConv_dgl(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(isize, isize))
            self.layers.append(GCNConv_dgl(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h, self.adj)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = process.knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities