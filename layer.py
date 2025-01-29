import dgl.function as fn
import torch
import torch.nn as nn

EOS = 1e-10


class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output


class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']


class Attentive(nn.Module):
    def __init__(self, isize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(isize))

    def forward(self, x):
        return x @ torch.diag(self.w)


class SparseDropout(nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - dprob

    def forward(self, x):
        mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(torch.bool)
        rc = x._indices()[:,mask]
        val = x._values()[mask]*(1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)


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
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
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