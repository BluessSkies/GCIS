import torch
import networkx as nx
import sys
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import copy
import numpy as np
from sklearn import metrics
from munkres import Munkres
import torch.nn.functional as F
import torch.nn as nn
from models import LogReg
from torch_scatter import scatter_add
import dgl
from kmeans_gpu import kmeans as kmeans_plusplus_gpu
import scipy.sparse as sp





def find_value_from_keys(dicts,key_list):
    value_list = []
    for key in key_list:
        value_list.append(dicts[key][0])
    return value_list


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro

class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):

        pred_label = self.pred_label
        # pred_label = new_predict
        acc = metrics.accuracy_score(self.true_label, pred_label)
        f1_macro = metrics.f1_score(self.true_label, pred_label, average='macro')
        precision_macro = metrics.precision_score(self.true_label, pred_label, average='macro')
        recall_macro = metrics.recall_score(self.true_label, pred_label, average='macro')
        f1_micro = metrics.f1_score(self.true_label, pred_label, average='micro')
        precision_micro = metrics.precision_score(self.true_label, pred_label, average='micro')
        recall_micro = metrics.recall_score(self.true_label, pred_label, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self, print_results=True):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        if print_results:
            print('ACC={:.4f}, f1_macro={:.4f}, precision_macro={:.4f}, recall_macro={:.4f}, f1_micro={:.4f}, '
                  .format(acc, f1_macro, precision_macro, recall_macro, f1_micro) +
                  'precision_micro={:.4f}, recall_micro={:.4f}, NMI={:.4f}, ADJ_RAND_SCORE={:.4f}'
                  .format(precision_micro, recall_micro, nmi, adjscore))

        return acc, nmi, f1_macro, adjscore

def clustering_lp(train_embs, train_lbls, nb_classes, kmeans_type):

    y_lbl = train_lbls.cpu().numpy()
    best_acc = 0.0
    if kmeans_type == 'e':
        for _ in range(5):
            y_pred_lp, _ = kmeans_lp(kmeans_type, nb_classes, train_embs)
            acc, f1 = cluster_acc(y_lbl, y_pred_lp)
            if acc > best_acc:
                best_acc = acc
    else:
        node_pred, _ = kmeans_lp(kmeans_type, nb_classes, train_embs)
        #best_acc, f1 = cluster_acc(node_pred, y_lbl)
    #print(best_acc, f1)
    return best_acc





def clu_center_node_sim(train_embs,cluster_centers):
    from sklearn.metrics.pairwise import cosine_similarity
    center_node_sim = cosine_similarity(train_embs.cpu().tolist(), cluster_centers.cpu().tolist())

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def pos_neg_adj(class_idx_dict, node_number):
    from itertools import product
    import networkx as nx

    edgelist_pos = []
    for (k, idx) in class_idx_dict.items():
        for i in range(len(idx)):
            for j in range(len(idx) - i - 1):
                j = j + i + 1
                edgelist_pos.append((idx[i], idx[j]))
    edgelist_neg = []
    for num in range(len(class_idx_dict)):
        dict1 = class_idx_dict[num]
        for num2 in range(len(class_idx_dict) - num - 1):
            dict2 = class_idx_dict[num2 + num + 1]
            result = product(dict1, dict2)
            for i in result:
                edgelist_neg.append(i)

    G = nx.Graph()
    G.add_nodes_from(range(2708))
    G.add_edges_from(edgelist_pos)
    adj_pos = np.array(nx.adjacency_matrix(G).todense())
    G = nx.Graph()
    G.add_nodes_from(range(2708))
    G.add_edges_from(edgelist_neg)
    adj_neg = np.array(nx.adjacency_matrix(G).todense())
    return torch.from_numpy(adj_pos).float(), torch.from_numpy(adj_neg).float()

def kmeans_sample_loss(train_fts, adj_pos, adj_neg):

    tau_knbrs = 1.0
    f = lambda x: torch.exp(x / tau_knbrs)
    matrix_sim = f(sim(train_fts, train_fts))

    pos_sum = torch.sum(torch.mul(matrix_sim, adj_pos))
    neg_sum = torch.sum(torch.mul(matrix_sim, adj_neg))

    loss1 = (-torch.log(pos_sum / (pos_sum+neg_sum)))
    return loss1


def kmeans_sample_loss2(train_fts, class_idx_dict):
    # tau_knbrs = 1.0
    # f = lambda x: torch.exp(x / tau_knbrs)
    # matrix_sim = f(sim(train_fts, train_fts))
    # pos_sum = torch.sum(torch.mul(matrix_sim, adj_pos))
    # neg_sum = torch.sum(torch.mul(matrix_sim, adj_neg))
    #
    # loss1 = (-torch.log(pos_sum / (pos_sum+neg_sum)))

    pos = torch.exp(torch.mm(train_fts, train_fts.t().contiguous())).cpu()
    nb_classes = len(class_idx_dict)
    pos_samp_sum = torch.zeros(pos.shape[0], 1)
    neg_samp_sum = torch.zeros(pos.shape[0], 1)

    for m in range(nb_classes):
        idx_now = torch.tensor(class_idx_dict[m])
        idx_all = idx_now
        for n in range(nb_classes):
            if n != m:
                idx_neg = torch.tensor(class_idx_dict[n])
                neg_samp = torch.zeros(pos.shape[0], len(idx_neg))
                neg_samp[idx_now] = pos.index_select(1, idx_neg)[idx_now]
                neg_samp_sum[idx_now] = neg_samp_sum[idx_now] \
                                        + neg_samp.sum(dim=-1)[idx_now].unsqueeze(1)
                idx_all = torch.cat((idx_all, idx_neg))
            else:
                idx = idx_now
                pos_samp = torch.zeros(pos.shape[0], len(idx))
                pos_samp[idx] = pos.index_select(1, idx)[idx]
                pos_samp_sum[idx] = pos_samp.sum(dim=-1)[idx].unsqueeze(1)
    pos_samp_sum = pos_samp_sum[idx_all].cuda()
    neg_samp_sum = neg_samp_sum[idx_all].cuda()
    node_contra_loss_2 = (- torch.log(pos_samp_sum / (pos_samp_sum + neg_samp_sum))).mean()

    return node_contra_loss_2


def kmeans_sample_loss3(train_fts, class_idx_dict):


    pos = torch.exp(sim(train_fts, train_fts))
    nb_classes = len(class_idx_dict)
    pos_samp_sum = torch.zeros(pos.shape[0], 1).cuda()
    neg_samp_sum = torch.zeros(pos.shape[0], 1).cuda()

    for m in range(nb_classes):
        idx_now = torch.tensor(class_idx_dict[m]).cuda()
        idx_all = idx_now
        for n in range(nb_classes):
            if n != m:
                idx_neg = torch.tensor(class_idx_dict[n]).cuda()
                neg_samp = torch.zeros(pos.shape[0], len(idx_neg)).cuda()
                neg_samp[idx_now] = pos.index_select(1, idx_neg)[idx_now]
                neg_samp_sum[idx_now] = neg_samp_sum[idx_now] \
                                        + neg_samp.sum(dim=-1)[idx_now].unsqueeze(1)
                idx_all = torch.cat((idx_all, idx_neg))
            else:
                idx = idx_now
                pos_samp = torch.zeros(pos.shape[0], len(idx)).cuda()
                pos_samp[idx] = pos.index_select(1, idx)[idx]
                pos_samp_sum[idx] = pos_samp.sum(dim=-1)[idx].unsqueeze(1)
    pos_samp_sum = pos_samp_sum[idx_all]
    neg_samp_sum = neg_samp_sum[idx_all]
    node_contra_loss_2 = (- torch.log(pos_samp_sum / (pos_samp_sum + neg_samp_sum))).mean()

    return node_contra_loss_2

def true_idx_dict(train_lbls, idx_train, nb_classes):

    idx_dict = dict()
    for i in range(nb_classes):
        idx_list = torch.masked_select(idx_train, train_lbls == torch.tensor(i)).tolist()
        idx_dict.update({i : idx_list})
    return idx_dict

from Bio.Cluster import kcluster, clustercentroids
from sklearn.cluster import KMeans
import torch
from kmeans_pytorch import kmeans



def kmeans_lp(kmeans_type, nb_classes, train_embs):



    if kmeans_type == "e":
        kmeans = KMeans(n_clusters=nb_classes, n_init=20)
        node_pred_lp = kmeans.fit_predict(train_embs)
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)
        # node_pred_lp, cluster_centers = kmeans(
        #     X=train_embs, num_clusters=nb_classes, distance='euclidean', device=torch.device('cuda')
        # )

    else:

        node_pred_lp, _, _ = kcluster(train_embs, nb_classes, dist='u', npass=10)
        cluster_centers, cmask = clustercentroids(train_embs, mask=None, transpose=0, clusterid=node_pred_lp)

    return node_pred_lp, cluster_centers
import numpy as np
import torch
from tqdm import tqdm


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

from tqdm import tqdm
def kmeans_plus_gpu(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)
    # initial_state = kmeans_plusplus_initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), initial_state.cpu()
def kmeans_plusplus_initialize(X, n_clusters, random_state=None):
    n_samples = X.shape[0]
    centers = torch.empty((n_clusters, X.shape[1]), dtype=X.dtype, device=X.device)

    if random_state is None:
        random_state = torch.Generator()

    # Choose the first center at random from the data points
    center_id = torch.randint(n_samples, (1,), generator=random_state)
    centers[0] = X[center_id]

    # Compute the distance from each data point to the first center
    closest_dist_sq = torch.norm(X - centers[0], dim=1) ** 2

    for c in range(1, n_clusters):
        # Choose the next center with probability proportional to the squared distance
        rand_vals = torch.multinomial(closest_dist_sq, 1, generator=random_state)
        centers[c] = X[rand_vals]

        # Compute the distance from each data point to the closest center
        dist = torch.norm(X[:, None] - centers[:c+1], dim=2)
        closest_dist_sq = torch.min(dist, dim=1).values ** 2

    return centers




def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis






def plot_louvain(G):
    import community.community_louvain as cl
    import networkx as nx
    import matplotlib.pyplot as plt

    # Replace this with your networkx graph loading depending on your format !
    #G = nx.erdos_renyi_graph(30, 0.05)
    #first compute the best partition
    partition = cl.best_partition(G)

    #drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                    node_color = str(count / size))


    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()



def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph

def cluster_pre_optimization(y_true, y_pred):

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    new_predict = list(new_predict.astype(int))
    return new_predict


def cluster_eval(train_embs,n_clu_trials,nb_classes,kmeans_type,train_lbls):
    
    acc = 0
    for clu_trial in range(n_clu_trials):
        predict_labels, _ = kmeans_lp(kmeans_type, nb_classes, train_embs)
        predict_labels = cluster_pre_optimization(list(train_lbls), list(predict_labels))

        cm_all = clustering_metrics(train_lbls, predict_labels)
        acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
        if acc_ >= acc:
            acc = acc_
            nmi = nmi_
            f1 = f1_
            ari = ari_
    return acc,nmi,f1,ari

def classfication_model(pse_train_embs,nb_classes,pse_lbl,edge_embs,edge_real_lbl):
    pse_train_embs = torch.tensor(pse_train_embs).cuda()
    edge_embs = torch.tensor(edge_embs).cuda()
    pse_lbl = torch.tensor(pse_lbl).cuda()
    edge_real_lbl = torch.tensor(edge_real_lbl).cuda()

    xent = nn.CrossEntropyLoss()
    for _ in range(50):
        log = LogReg(pse_train_embs.shape[1], nb_classes)
        # 线性层，从512转为7
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(pse_train_embs)

            loss = xent(logits, pse_lbl)

            loss.backward()
            opt.step()

        logits = log(edge_embs)
        edge_pred_lbl = torch.argmax(logits, dim=1)

        acc = torch.sum(edge_pred_lbl == edge_real_lbl).float() / edge_real_lbl.shape[0]

    return edge_pred_lbl,acc



def get_node_dist(graph):
    """
    Compute adjacent node distribution.
    获取2708*2708的边分布
    """
    row, col = graph.edges()[0], graph.edges()[1]
    num_node = graph.num_nodes()

    dist_list = []
    for i in range(num_node):
        dist = torch.zeros([num_node], dtype=torch.float32, device=graph.device)
        idx = row[(col==i)]
        dist[idx] = 1
        dist_list.append(dist)
    dist_list = torch.stack(dist_list, dim=0)
    return dist_list

def get_sim(embeds1, embeds2):
    # normalize embeddings across feature dimension
    embeds1 = F.normalize(embeds1)
    embeds2 = F.normalize(embeds2)
    sim = torch.mm(embeds1, embeds2.t())
    return sim


def degree_mask_edge(idx, sim, max_degree, node_degree, mask_prob):
    aug_degree = (node_degree * (1- mask_prob)).long().to(sim.device)
    sim_dist = sim[idx]

    # _, new_tgt = th.topk(sim_dist + 1e-12, int(max_degree))
    new_tgt = torch.multinomial(sim_dist + 1e-12, int(max_degree))
    tgt_idx = torch.arange(max_degree).unsqueeze(dim=0).to(sim.device)

    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = idx.repeat_interleave(aug_degree)
    return new_row, new_col
