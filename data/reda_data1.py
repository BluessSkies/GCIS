import numpy as np
import pickle as pkl
import sys
import torch
from scipy import sparse


objects = []
with open("ind.cora.x", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))
x = objects[0].toarray()

objects = []
with open("ind.cora.tx", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))
tx = objects[0].toarray()
sstx = objects[0]
objects = []
with open("ind.cora.allx", 'rb') as f:
    if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding='latin1'))
    else:
        objects.append(pkl.load(f))

allx = objects[0].toarray()


stx = sparse.csr_matrix(tx)

def load_graph_data(dataset_name, show_details=False):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """
    dataset_name = 'amap'
    load_path = "dataset/" +  dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    # X pre-processing
    pca = PCA(n_components=opt.args.n_input)
    feat = pca.fit_transform(feat)
    return feat, label, adj