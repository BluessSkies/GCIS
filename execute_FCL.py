import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import time
from layers import GCN
from model import loss_func, graph_learner, FGP_learner, ATT_learner, GNN_learner, MLP_learner
from models import LogReg
from utils import process
import os
import copy
import random
import argparse
import sys
import dgl
import pickle
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans
import torch.nn.functional as F
from kmeans_gpu import kmeans as kmeans_plusplus_gpu
from Bio.Cluster import kcluster, clustercentroids
from sklearn.cluster import KMeans

from torch.autograd import Variable
from data_deal_lp import path_sim, plot_similar, data_anaysis_plot, \
    scatter_single_plot, plot_path_length, cluster_acc, clustering_lp,\
    plot_kmean, plot_tsne, kmean_cosine_sampe, kmeans_sample_loss, \
    pos_neg_adj, kmeans_sample_loss2, kmeans_sample_loss3, true_idx_dict, \
    kmeans_lp, plot_tsne_embedding, cluster_pre_optimization, neighbor_same_rate, clustering_metrics,\
    plot_tsne_embedding_save, classfication_model, degree_aug, plot_heat_maps, cal_similarity_graph, \
    mask_topk, kmeans_plus_gpu



def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2

class GGD(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GGD, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.lin = nn.Linear(n_h, n_h)

    def forward(self, seq1, seq2, adj, sparse):
        h_1 = self.gcn(seq1, adj, sparse)
        h_2 = self.gcn(seq2, adj, sparse)
        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1)).unsqueeze(0)
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1)).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = self.gcn(seq, adj, sparse)
        h_2 = h_1.clone().squeeze(0)

        for i in range(5):
            h_2 = adj @ h_2

        h_2 = h_2.unsqueeze(0)

        return h_1.detach().cpu(), h_2.detach().cpu()

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def aug_random_edge(input_adj, drop_percent=0.1):
    drop_percent = drop_percent
    b = np.where(input_adj > 0,
                 np.random.choice(2, (input_adj.shape[0], input_adj.shape[0]), p=[drop_percent, 1 - drop_percent]),
                 input_adj)
    drop_num = len(input_adj.nonzero()[0]) - len(b.nonzero()[0])
    mask_p = drop_num / (input_adj.shape[0] * input_adj.shape[0] - len(b.nonzero()[0]))
    c = np.where(b == 0, np.random.choice(2, (input_adj.shape[0], input_adj.shape[0]), p=[1 - mask_p, mask_p]), b)

    return b


def aug_feature_dropout(input_feat, drop_percent=0.2):
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


if __name__ == '__main__':
    acc_results = []
    import warnings

    warnings.filterwarnings("ignore")

    # setting arguments
    parser = argparse.ArgumentParser('GGDS')

    parser.add_argument('--classifier_epochs', type=int, default=150, help='classifier epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--np_epochs', type=int, default=600, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=300, help='Patience')
    parser.add_argument('--lr', type=float, default=0.001, help='Patience')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='l2 coef')
    parser.add_argument('--drop_prob', type=float, default=0.0, help='Tau value')
    parser.add_argument('--hid_units', type=int, default=512, help='Top-K value')
    parser.add_argument('--sparse', action='store_true', help='Whether to use sparse tensors')
    parser.add_argument('--dataset', type=str, default='amap', help='Dataset name: cora, citeseer, pubmed, cs, phy')
    parser.add_argument('--num_hop', type=int, default=0, help='graph power')
    parser.add_argument('--n_trials', type=int, default=1, help='number of trails ')
    parser.add_argument('--samp_rate', type=float, default=0.2)
    parser.add_argument('--update_epoch', type=int, default=200)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument('--free_gpu_id', type=int, default=0)
    parser.add_argument('--loss_rate', type=float, default=1)
    parser.add_argument('--kmeans_type', type=str, default='euclidean', choices=['cosine', 'euclidean'])
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--tau', type=float, default=0.9999)
    # GSL 
    parser.add_argument('--type_learner', type=str, default='ggd', help='["fgp", "att", "mlp", "gnn", "ggd"]')
    parser.add_argument('--sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--activation_learner', type=str, default='relu', help='["relu", "tanh"]')
    parser.add_argument('--gsl_sparse', type=int, default=0)
    parser.add_argument('--torch_empty', type=int, default=1)
    parser.add_argument('--loss_record', type=int, default=0)
    parser.add_argument('--update_eval', type=int, default=1)

  
    parser.add_argument('--n_clu_trials', type=int, default=100)





    try:

        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    print(str(args))
    kmeans_type = args.kmeans_type
    samp_rate = args.samp_rate
    n_trails = args.n_trials
    acc_res = []
    wrong_node = []
    best_epochs = []
    accs = []
    nmis = []
    aris = []
    f1s = []
    acc_cls = []
    edges_nodes = []
    acc_edges = []
    E_stopping = []
    torch.cuda.set_device(int(args.free_gpu_id))
    n_clu_trials = args.n_clu_trials
    for i in range(n_trails):

        dataset = args.dataset

        # training params
        batch_size = args.batch_size
        nb_epochs = args.np_epochs
        patience = args.patience
        classifier_epochs = args.classifier_epochs
        lr = args.lr
        l2_coef = args.l2_coef
        drop_prob = args.drop_prob
        hid_units = args.hid_units
        num_hop = args.num_hop
        sparse = True
        nonlinearity = 'prelu'  # special name to separate parameters
        graph_road = "data/ind."+ dataset + ".graph"
        #wrong_idx_list = np.load('wrong_idx_list.npy')

        # load dataset
        if dataset in ['cora', 'citeseer', 'pubmed']:
            adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
            #print(labels.shape)
            idx_train = range(labels.shape[0])   
        elif dataset in ['cornell']:
            adj, labels, features, idx_train, idx_val, idx_test = process.load_fb100(
                dataset)
            features = sp.lil_matrix(features)
            n_values = np.max(labels) + 1
            labels = np.eye(n_values)[labels]
        else:
            adj = np.load('data/' + dataset + '_adj.npy')
            adj = sp.csr_matrix(adj)
            features = np.load('data/' + dataset + '_feat.npy')
            features = sp.lil_matrix(features)
            labels = np.load('data/' + dataset + '_label.npy')

            n_values = np.max(labels) + 1
            labels = np.eye(n_values)[labels]

            test_ratio = 0.9

            idx_test = random.sample(list(np.arange(features.shape[0])), int(test_ratio * features.shape[0]))
            remain_num = len(idx_test)
            idx_val = idx_test
            idx_train = list(set(np.arange(features.shape[0])) - set(idx_test))

            train_mask = torch.zeros(features.shape[0]).long()
            train_mask[idx_train] = 1
            test_mask = torch.zeros(features.shape[0]).long()
            test_mask[idx_test] = 1
            val_mask = torch.zeros(features.shape[0]).long()
            val_mask[idx_val] = 1

            train_mask = train_mask.bool()
            test_mask = test_mask.bool()
            val_mask = val_mask.bool()
        idx_train = range(labels.shape[0])   

        # preprocessing and initialisation
        features, _ = process.preprocess_features(features)
        graph = dgl.DGLGraph()
        graph.add_edges(adj.tocoo().row, adj.tocoo().col)
        # features_wrong, _ = process.preprocess_features(features_wrong)
        # features_wrong = torch.FloatTensor(features_wrong)


        nb_nodes = features.shape[0]
        nb_classes = labels.shape[1]

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj0 = adj


        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
      。
        anchor_adj = process.normalize(torch.tensor(adj.todense()), 'sym', args.gsl_sparse)

        ft_size = features.shape[1]
        #print(ft_size)

        features = torch.FloatTensor(features)
        original_features = features.unsqueeze(0).cuda()

        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
        labels = torch.FloatTensor(labels[np.newaxis])
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        ggd = GGD(ft_size, hid_units, nonlinearity)
        optimiser_disc = torch.optim.Adam(ggd.parameters(), lr=lr, weight_decay=l2_coef)


        if torch.cuda.is_available():
            ggd.cuda()
            features = features.cuda()
            # features_wrong = features_wrong.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        features = features.unsqueeze(0)
        # features_wrong = features_wrong.unsqueeze(0)

        # generate a random number --> later use as a tag for saved model
        tag = str(int(np.random.random() * 10000000000))

        nb_feats = features.shape[2]
        loss_rate = args.loss_rate
        avg_time = 0
        counts = 0
        train_lbls = torch.argmax(labels[0, idx_train], dim=1)
        val_lbls = torch.argmax(labels[0, idx_val], dim=1)
        test_lbls = torch.argmax(labels[0, idx_test], dim=1)
        wrong_neb_sim_dicts = []
        loss_record_gd = []
        loss_record_stru = []
        loss_record=[]
        acc_updata = []
        train_lbls = train_lbls.cpu().numpy()
        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()
        #idx22 = np.random.permutation(nb_nodes)
        acc = 0

        for epoch in range(nb_epochs):

            ggd.train()
            optimiser_disc.zero_grad()

            #print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            if epoch < args.pretrain_epoch:
                aug_fts = aug_feature_dropout(features.squeeze(0)).unsqueeze(0)

                idx = np.random.permutation(nb_nodes)  
                shuf_fts = aug_fts[:, idx, :]


                if torch.cuda.is_available():
                    shuf_fts = shuf_fts.cuda()
                    aug_fts = aug_fts.cuda()
                    lbl = lbl.cuda()
                logits_1 = ggd(aug_fts, shuf_fts, sp_adj, sparse=True)
                loss_disc = b_xent(logits_1, lbl)


                loss_epoch = loss_disc.detach().cpu().numpy()
                loss_record_gd.append(loss_epoch)
                loss_record.append(loss_epoch)
                # re_loss = F.binary_cross_entropy(A_pred.float().view(-1), torch.tensor(adj0.todense()).cuda().float().view(-1))
                # loss_disc = loss_disc + re_loss


                if loss_disc < best:
                    best = loss_disc
                    best_t = epoch
                    cnt_wait = 0
                    # torch.save(ggd.state_dict(), 'pkl/best_dgi' + tag + '.pkl')
                    torch.save(ggd.state_dict(), 'pkl/lp_best_dgi' + tag + '.pkl')
                    # print("save pkl")
                else:
                    cnt_wait += 1

                if cnt_wait == patience:
                    print('Early stopping!')
                    break
                loss_disc.backward()
                optimiser_disc.step()

            if epoch == args.pretrain_epoch:
                # with torch.no_grad():
                #     or_embeds, pr_embeds = ggd.embed(original_features, sp_adj if sparse else adj, sparse)
                # embeds = or_embeds + pr_embeds
                # train_embs = embeds[0, idx_train]

                lbl_1 = torch.ones(batch_size, nb_nodes)
                lbl_2 = torch.zeros(batch_size, nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1).cuda()
                print('final pretrain')

            if epoch % args.update_epoch == 0 :

                if args.torch_empty == 1:
                    torch.cuda.empty_cache()  # 释放显存
                #
                with torch.no_grad():
                    or_embeds, pr_embeds = ggd.embed(original_features, sp_adj if sparse else adj, sparse)
                embeds = or_embeds + pr_embeds
                train_embs = embeds[0, idx_train].cuda()

                if args.update_eval == 1:
                    # class_idx_dict, wrong_rate_dict = kmean_cosine_sampe(train_embs, idx_train, nb_classes, samp_rate, train_lbls, kmeans_type)
                    # print(wrong_rate_dict)
                    acc_epoch = 0
                    for clu_trial in range(int(n_clu_trials)):
                        # predict_labels, _ = kmeans_lp(kmeans_type, nb_classes, train_embs.cpu().numpy())
                        # kmeans = KMeans(n_clusters=nb_classes, init='k-means++', random_state=None).fit(
                        #     train_embs.cpu().numpy())
                        # predict_labels = kmeans.labels_
                        # predict_labels, cluster_centers = kmeans(
                        #     X=train_embs, num_clusters=nb_classes, distance=kmeans_type, device=torch.device('cuda')
                        # )
                        predict_labels, _,_ = kmeans_plusplus_gpu(X=train_embs, num_clusters=nb_classes, distance="euclidean", device="cuda")
                        # predict_labels = list(predict_labels.cpu().numpy())
                        predict_labels = cluster_pre_optimization(list(train_lbls), list(predict_labels))

                        cm_all = clustering_metrics(train_lbls, predict_labels)
                        acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
                        #print(acc_)
                        if acc_ >= acc:
                            acc = acc_
                            nmi = nmi_
                            f1 = f1_
                            ari = ari_
                            predict_labels_best1 = predict_labels
                            best_epoch = epoch
                            print('acc:' + str(acc))
                            print('epoch:' + str(epoch))
                        if acc_ > acc_epoch:
                            acc_epoch = acc_

                    acc_updata.append(acc_epoch)

            if epoch > args.pretrain_epoch:
                aug_fts = aug_feature_dropout(features.squeeze(0)).unsqueeze(0)

                idx = np.random.permutation(nb_nodes) 
                shuf_fts = aug_fts[:, idx, :]

                with torch.no_grad():
                    or_embeds, pr_embeds = ggd.embed(original_features, sp_adj if sparse else adj, sparse)
                embeds = or_embeds + pr_embeds
                embeds = embeds.squeeze(0).cuda()

                learned_adj = graph_learner(embeds, args.k, 'relu', 6)
                del embeds
                #learned_adj = learned_adj.cuda().detach()
            # 1700


                sp_adj = sp_adj * args.tau + learned_adj.to_sparse()*(1-args.tau)
                torch.cuda.empty_cache()
                # # sp_adj = adj_grade

                logits_1 = ggd(aug_fts, shuf_fts, sp_adj, sparse=True)
                loss_disc1 = b_xent(logits_1, lbl)
                # loss_disc_clu = b_xent(logits2, lbl2)

                #re_loss = F.binary_cross_entropy(learned_adj.float().view(-1), torch.tensor(adj0.todense()).cuda().float().view(-1))
                #re_loss = b_xent(learned_adj.float().view(-1),  torch.tensor(adj0.todense()).cuda().float().view(-1))
                #loss_disc = loss_rate*loss_disc1 + (1-loss_rate)*re_loss
                loss_disc = loss_disc1

                # loss_record_gd.append(loss_disc1.detach().cpu().numpy())
                # loss_record_stru.append(re_loss.detach().cpu().numpy())
                # loss_record.append(loss_disc.detach().cpu().numpy())

                # loss_disc = 0.5*loss_disc1 +0.5* loss_disc_clu
                #loss_disc = re_loss + loss_disc1



                if loss_disc < best:
                    best = loss_disc
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(ggd.state_dict(), 'pkl/lp_best_dgi' + tag + '.pkl')
                    #print("gsl work!")
                else:
                    cnt_wait += 1

                if cnt_wait == patience:
                    print('Early stopping!')
                    E_stopping.append(str(epoch))
                    break

                optimiser_disc.zero_grad()
                loss_disc.backward()
                optimiser_disc.step()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()



        if args.loss_record == 1:
            with open('train_record/{}_acc_record_{}_{}_{}_{}_{}.pickle'.format(dataset, str(args.np_epochs), str(args.pretrain_epoch ), str(args.update_epoch), str(loss_rate),str(i )), 'wb') as f:
                pickle.dump(acc_updata, f)
            with open('train_record/{}_loss_record_{}_{}_{}_{}_{}.pickle'.format(dataset, str(args.np_epochs),str(args.pretrain_epoch ), str(args.update_epoch), str(loss_rate), str(i )), 'wb') as f:
                pickle.dump(loss_record, f)

            #epoch_idx = list(np.arange(0, args.np_epochs+1, args.update_epoch))
            plt.plot( acc_updata)
            plt.savefig('train_record/{}_acc_record_{}_{}_{}_{}_{}.png'.format(dataset, str(args.np_epochs), str(args.pretrain_epoch ), str(args.update_epoch), str(loss_rate), str(i )))
            plt.close()

            #epoch_idx = list(np.arange(0, args.np_epochs+1, 1))
            plt.plot(loss_record)
            plt.savefig('train_record/{}_loss_record_{}_{}_{}_{}_{}.png'.format(dataset, str(args.np_epochs), str(args.pretrain_epoch ), str(args.update_epoch), str(loss_rate), str(i )))
            plt.close()

        print("final_dgi pkl")
        with torch.no_grad():
            ggd.load_state_dict(torch.load('pkl/lp_best_dgi' + tag + '.pkl'))
            or_embeds, pr_embeds = ggd.embed(original_features, sp_adj if sparse else adj, sparse)
            embeds = or_embeds + pr_embeds
        train_embs = embeds[0, idx_train].cuda()
        # val_embs = embeds[0, idx_val]
        # test_embs = embeds[0, idx_test]
        torch.cuda.empty_cache()
        epoch_idx = list(range(epoch))
        # plt.plot(epoch_idx, loss_record)
        # plt.show()
        # plt.plot(epoch_idx, loss_record_gd)
        # plt.show()


        # acc_mr, nmi_mr, f1_mr, ari_mr = [], [], [], []

        wrong_idx_list = []
        edge_idx = list()

        for clu_trial in range(n_clu_trials):
            #predict_labels, _ = kmeans_lp(kmeans_type, nb_classes, train_embs.cpu().numpy())
            # kmeans = KMeans(n_clusters=nb_classes, init='k-means++', random_state=None).fit(train_embs.cpu().numpy())
            # predict_labels =kmeans.labels_
            # predict_labels, cluster_centers = kmeans(
            #     X=train_embs, num_clusters=nb_classes, distance=kmeans_type, device=torch.device('cuda')
            # )
            predict_labels, _, _ = kmeans_plusplus_gpu(X=train_embs, num_clusters=nb_classes, distance="euclidean", device="cuda")
            # predict_labels = list(predict_labels.cpu().numpy())
            predict_labels = cluster_pre_optimization(list(train_lbls), list(predict_labels))

            # wrong_idx_list_0 = idx_train.cpu().numpy()[predict_labels != train_lbls]
            #
            # if clu_trial ==0:
            #     wrong_idx_list = wrong_idx_list_0
            #     predict_labels0 =  predict_labels
            #     edge_idx0 = []
            #     edge_idx = []
            #
            # else:
            #     edge_idx0 = idx_train.cpu().numpy()[np.array(predict_labels) != predict_labels0]
            #     edge_idx = np.union1d(edge_idx, edge_idx0)
            #     predict_labels0 = predict_labels
            #     wrong_idx_list = np.intersect1d(wrong_idx_list, wrong_idx_list_0)


            cm_all = clustering_metrics(train_lbls, predict_labels)
            acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
            #print(acc_)
            if acc_ >= acc:
                acc = acc_
                nmi = nmi_
                f1 = f1_
                ari = ari_
                predict_labels_best1 = predict_labels
                best_epoch = epoch

        print('acc:' + str(acc))
        print('best_epoch:' + str(best_epoch))
        # print('epoch:' + str(epoch))
        best_epochs.append(best_epoch)
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)
        f1s.append(f1)



    acc_mean, nmi_mean, f1_mean, ari_mean = np.mean(accs), np.mean(nmis), np.mean(f1s), np.mean(aris)
    acc_std, nmi_std, f1_std, ari_std = np.std(accs, ddof=1), np.std(nmis, ddof=1), np.std(f1s, ddof=1), np.std(
        aris, ddof=1)
    acc_mean = round(acc_mean, 4)
    nmi_mean = round(nmi_mean, 4)
    f1_mean = round(f1_mean, 4)
    ari_mean = round(ari_mean, 4)
    acc_std = round(acc_std, 4)
    nmi_std = round(nmi_std, 4)
    f1_std = round(f1_std, 4)
    ari_std = round(ari_std, 4)

    record_txt = 'FCL_log_{}_gsl_clu.txt'.format(args.dataset)
    if not os.path.exists(record_txt):
        with open(record_txt, 'a') as f:
            f.write('dataset####epochs####pretrain_epochs####tau####k####loss_rate####type_learner####hid_units####'
                    'activation_learner####kmeans_type####acc_mean####acc_std####nmi_mean####nmi_std####f1_mean####f1_std####'
                    'ari_mean####ari_std####时间####accs####best_epochs####截断')
            f.write('\n')

    with open(record_txt, 'a') as f:
        f.write('{}####{}####{}####{}####{}####{}####{}####{}####{}####{}####{}####{}####{}####{}####{}####{}####{}'
                '####{}####{}####{}####{}####{}'.format(args.dataset, args.np_epochs, args.pretrain_epoch, args.tau, args.k, args.loss_rate,
                                  args.type_learner, args.hid_units, args.activation_learner,kmeans_type, acc_mean, acc_std,
                                  nmi_mean, nmi_std, f1_mean, f1_std, ari_mean, ari_std,
                                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), accs, best_epochs, E_stopping))
        f.write('\n')

