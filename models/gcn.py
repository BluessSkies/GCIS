import torch
import torch.nn as nn
from .constants import VERY_SMALL_NUMBER


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False, pivot_mp = False):
        seq_fts = self.fc(seq)
        if pivot_mp:
            node_pivot_adj = adj
            node_norm = node_pivot_adj / \
                torch.clamp(torch.sum(node_pivot_adj, dim=-2,
                            keepdim=True), min=VERY_SMALL_NUMBER)
            pivot_norm = node_pivot_adj / \
                torch.clamp(torch.sum(node_pivot_adj, dim=-1,
                            keepdim=True), min=VERY_SMALL_NUMBER)
            out = torch.matmul(pivot_norm, torch.matmul(
                node_norm.transpose(-1, -2), seq_fts))

        else:
            if sparse:
                # out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
                out = torch.spmm(adj, torch.squeeze(seq_fts, 0))
                # Torch.spmm只支持 sparse 在前，dense 在后的矩阵乘法
            else:
                out = torch.bmm(adj, seq_fts)
                # torch.bmm 计算两个tensor的矩阵乘法
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class GCN_m(nn.Module):
    def __init__(self, in_ft, out_ft, act, k, bias=True):
        self.act = nn.PReLU() if act == 'prelu' else act
        self.conv = [GCN(in_ft, out_ft)]
        for _ in range(0, k):
            self.conv.append(GCN(out_ft, out_ft))
        self.conv = nn.ModuleList(self.conv)