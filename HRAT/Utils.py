import os
# from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix


import numpy as np
import torch
# import faiss

import argparse

def parse_arguments():
    """
    Function to initialize the argument parser and parse command-line arguments.

    Returns:
    --------
    args : argparse.Namespace
        Parsed arguments from the command line.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Parse input datasets and save directory.")

    # Define the required arguments
    parser.add_argument('--train_set', type=str, required=True, help="Path to the training dataset.")
    parser.add_argument('--test_set', type=str, required=True, help="Path to the test dataset.")
    parser.add_argument('--test_set_cons', type=str, required=True, help="Constraints file for the test graph set.")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the results.")

    parser.add_argument('--attack_mlp', type=str, help="Path to the MLP model for the attack.")
    # Parse arguments from the command line
    return parser.parse_args()
def degree_centrality_extraction(adjacent_matrix, sen_idx):
    centrality = (adjacent_matrix.sum(axis=0) + adjacent_matrix.sum(axis=1).transpose()) / (
        adjacent_matrix.shape[0] - 1)

    centrality = np.array(centrality)
    centrality = np.squeeze(centrality)
    idx_matrix = np.zeros((len(sen_idx), adjacent_matrix.shape[0]))
    ii = np.where(sen_idx != -1)
    idx_matrix[ii, sen_idx[ii]] = 1
    feature = np.matmul(idx_matrix, centrality)
    return feature


def katz_feature(graph, sen_idx, alpha=0.1, beta=1.0, normalized=True, weight=None):
    graph = graph.T
    n = graph.shape[0]
    b = np.ones((n, 1)) * float(beta)
    centrality = np.linalg.solve(np.eye(n, n) - (alpha * graph), b)
    if normalized:
        norm = np.sign(sum(centrality)) * np.linalg.norm(centrality)
    else:
        norm = 1.0
    centrality = centrality / norm
    idx_matrix = np.zeros((len(sen_idx), n))
    ii = np.where(sen_idx != -1)
    idx_matrix[ii, sen_idx[ii]] = 1
    feature = np.matmul(idx_matrix, centrality)
    return feature


def trans2triple_rw(adjacent_matrix, sha256, triple_path,  overwrite=False):
    file_name = triple_path+"/" + sha256 + ".npy"
    triple = []

    if os.path.exists(file_name) and not overwrite:
        print("loading")
        triple = np.load(file_name)
        if triple.shape[0] < adjacent_matrix.shape[0]:
            triple = trans2triple_rw(
                adjacent_matrix, sha256, triple_path,  overwrite=True)
    else:
        node_number = adjacent_matrix.shape[0]
        triple = []
        # if type(adjacent_matrix) is sparse.coo_matrix:
        #     adjacent_matrix = adjacent_matrix.tocsr()
        if isinstance(adjacent_matrix, coo_matrix):
            adjacent_matrix = adjacent_matrix.tocsr()
        for zi in range(node_number):
            # triple.append([zi, zi, adjacent_matrix[zi, zi]])
            for zj in range(zi + 1, node_number):
                triple.append([zi, zj, adjacent_matrix[zi, zj]])
                triple.append([zj, zi, adjacent_matrix[zj, zi]])
        triple = np.array(triple)
        np.save(file_name, triple)
    return triple


#NN找邻居
# Q是查询点
# X：数据集中的点集合，通常是一个矩阵，每一行是一个点
# y：X中每个点的标签。
# k：要寻找的最近邻居的数量，默认为1
def find_nn_torch(Q, X, y, k=1, device='cuda'):
    # 函数逻辑：
    # （1）计算距离：
    # 计算查询点Q与数据集X中所有点之间的欧几里得距离的平方。计算结果存储在dist中。
    #
    # （2）排序距离：
    # 对距离进行排序，并获取排序后的索引ind。
    #
    # （3）获取最近邻标签：
    # 使用排序后的索引ind获取前k个最近邻居的标签，并存储在label中。
    #
    # （4）计算最近的良性样本距离：
    # 找到最近的良性样本（标签为1）的距离min_dist。
    #
    # （5）计算最频繁的标签：
    # 计算前k个最近邻居中出现最频繁的标签final_label。
    #
    # （6）返回结果：
    # 返回最频繁的标签final_label和最近的良性样本距离min_dist。
    # dist = torch.sum((np.squeeze(X) - np.squeeze(Q)).pow(2.), 1)#计算两点之间的欧几里德距离平方
    # ind = torch.argsort(dist)#排序，返回从小到大的点的下标
    # label = y[ind[:k]]# [:k]取前k个，y[]是从y张量中，找到对应的label
    #
    # label_total = y[ind]
    # list_label = label_total.cpu().numpy() #将label_total移到CPU，然后转成numpy数组
    # benign_idx = np.argwhere(list_label==1) #找到所有良性label的下标
    # min_dist = dist[ind[benign_idx[0][0]]] #初始化 最小的距离为第一个良性的下标
    #
    # unique_label = torch.unique(y)
    # unique_label = unique_label.long()
    # count = np.zeros(unique_label.shape[0])
    # for i in label:
    #     count[unique_label[i.long()]] += 1
    # ii = torch.argmax(torch.from_numpy(count))
    # final_label = unique_label[ii]
    # return final_label, min_dist
    # 确保所有张量都在同一设备上
    Q = Q.to(device)
    X = X.to(device)
    y = y.to(device)

    # 计算距离
    dist = torch.sum((torch.squeeze(X) - torch.squeeze(Q)).pow(2.), dim=1)

    # 排序距离
    ind = torch.argsort(dist)

    # 获取最近邻标签
    label = y[ind[:k]]

    # 获取最近的良性样本距离
    benign_idx = torch.where(y[ind] == 1)[0]
    min_dist = dist[ind[benign_idx[0]]] if benign_idx.size(0) > 0 else torch.tensor(float('inf'), device=device)

    # 计算最频繁的标签
    unique_label, counts = torch.unique(y[ind[:k]], return_counts=True)
    final_label = unique_label[torch.argmax(counts)]
    # print("final_label", final_label)
    # final_label = final_label.item()
    # print("final_label", final_label)

    return final_label, min_dist


def to_adjmatrix(adj_sparse, adj_size):
    A = torch.sparse_coo_tensor(adj_sparse[:, :2].T, adj_sparse[:, 2],
                                size=[adj_size, adj_size]).to_dense()
    return A


def degree_centrality_torch(adj, sen_api_idx, device='cuda'):
    adj_size = adj.shape[0]
    print("adj_size", adj_size)
    # idx_matrix = np.zeros((len(sen_api_idx), adj_size))
    # ii = np.where(sen_api_idx != -1)
    # idx_matrix[ii, sen_api_idx[ii]] = 1
    # idx_matrix = torch.from_numpy(idx_matrix).to(device)
    #
    # if adj.shape[0] > len(idx_matrix):
    #     _sub = adj.shape[0] - len(idx_matrix)
    #     for za in range(_sub):
    #         idx_matrix.append[0]
    #
    # all_degree = torch.div((torch.sum(adj, 0) + torch.sum(adj, 1)).float(),
    #                        float(adj.shape[0] - 1))
    # degree_centrality = torch.matmul(
    #     idx_matrix, all_degree.type_as(idx_matrix))
    # return degree_centrality
    idx_matrix = torch.zeros((len(sen_api_idx), adj_size), dtype=torch.float32).to(device)

    # 使用PyTorch操作替代np.where
    idx_matrix[sen_api_idx != -1, sen_api_idx[sen_api_idx != -1]] = 1

    adj_dense = to_adjmatrix(adj, adj_size)
    print("check adj_dense")
    print(adj_dense.shape)

    # 用PyTorch张量操作替换循环
    if adj_dense.shape[0] > idx_matrix.shape[0]:
        _sub = adj_dense.shape[0] - idx_matrix.shape[0]
        idx_matrix = torch.cat([idx_matrix, torch.zeros((_sub, adj_size), dtype=torch.float32).to(device)],
                               dim=0)

    # 使用PyTorch操作计算度中心性
    all_degree = torch.div((torch.sum(adj_dense, 0) + torch.sum(adj_dense, 1)).float(), adj_dense.shape[0] - 1)
    degree_centrality = torch.matmul(idx_matrix, all_degree.type_as(idx_matrix))

    return degree_centrality


def katz_feature_torch(graph, sen_api_idx, alpha=0.025, beta=1.0, device='cuda', normalized=True):
    n = graph.shape[0]
    graph = graph.T
    b = torch.ones((n, 1)) * float(beta)
    b = b.to(device)
    graph = graph.to(device)
    A = torch.eye(n, n).to(device).float() - (alpha * graph.float())
    # L, U = torch.solve(b, A)
    L = torch.linalg.solve(A, b)
    if normalized:
        norm = torch.sign(sum(L)) * torch.norm(L)
    else:
        norm = 1.0
    centrality = torch.div(L, norm.to(device)).to(device)
    idx_matrix = np.zeros((len(sen_api_idx), n))
    ii = np.where(sen_api_idx != -1)
    idx_matrix[ii, sen_api_idx[ii]] = 1
    idx_matrix = torch.from_numpy(idx_matrix).to(device)
    katz_centrality = torch.matmul(idx_matrix, centrality.type_as(idx_matrix))
    return katz_centrality

def closeness_centrality_torch(adj, sen_api_idx, device='cuda'):
    n = adj.shape[0]
    adj = adj.to(device)
    idx_matrix = np.zeros((len(sen_api_idx), n))
    ii = np.where(sen_api_idx != -1)
    idx_matrix[ii, sen_api_idx[ii]] = 1
    idx_matrix = torch.from_numpy(idx_matrix).to(device)

    # 使用 Floyd-Warshall 算法计算所有节点对之间的最短路径
    floyd_matrix = torch.full((n, n), float('inf')).to(device)
    floyd_matrix[torch.arange(n), torch.arange(n)] = 0
    floyd_matrix[adj != 0] = adj[adj != 0].float()

    for k in range(n):
        floyd_matrix = torch.min(floyd_matrix, floyd_matrix[:, k].unsqueeze(1) + floyd_matrix[k, :].unsqueeze(0))

    closeness_centrality = torch.div(n - 1, torch.sum(floyd_matrix, dim=1))
    return torch.matmul(idx_matrix, closeness_centrality.type_as(idx_matrix))

def harmonic_centrality_torch(adj, sen_api_idx, device='cuda'):
    n = adj.shape[0]
    adj = adj.to(device)
    idx_matrix = np.zeros((len(sen_api_idx), n))
    ii = np.where(sen_api_idx != -1)
    idx_matrix[ii, sen_api_idx[ii]] = 1
    idx_matrix = torch.from_numpy(idx_matrix).to(device)

    # 使用 Floyd-Warshall 算法计算所有节点对之间的最短路径
    floyd_matrix = torch.full((n, n), float('inf')).to(device)
    floyd_matrix[torch.arange(n), torch.arange(n)] = 0
    floyd_matrix[adj != 0] = adj[adj != 0].float()

    for k in range(n):
        floyd_matrix = torch.min(floyd_matrix, floyd_matrix[:, k].unsqueeze(1) + floyd_matrix[k, :].unsqueeze(0))

    # 防止除以零
    floyd_matrix[floyd_matrix == float('inf')] = 0
    harmonic_centrality = torch.div(1.0, floyd_matrix).sum(dim=1)
    return torch.matmul(idx_matrix, harmonic_centrality.type_as(idx_matrix))





def obtain_sensitive_apis(file):
    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            else:
                sensitive_apis.append(line.strip())
    return sensitive_apis


def extract_sensitive_api(sensitive_api_list, nodes_list):
    sample_sensitive_api = []
    for x in sensitive_api_list:
        if x in nodes_list:
            sample_sensitive_api.append(nodes_list.index(x))
        else:
            sample_sensitive_api.append(-1)
    return np.array(sample_sensitive_api)


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)



def check_folder(folder_name: str):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def fcg_to_adjacent(node_file, fcg_file):
    tmp_node = open(node_file, "r", encoding='utf-8').readlines()
    node_list = [zn.replace("\n", "") for zn in tmp_node]
    tmp_graph = open(fcg_file, "r", encoding='utf-8')
    line = tmp_graph.readline()
    row_ind = []
    col_ind = []
    data = []
    while line:
        # print(line)
        line = line.split("\n")[0]
        nodes = line.split(" ==> ")
        row_ind.append(node_list.index(nodes[0]))
        col_ind.append(node_list.index(nodes[1]))
        data.append(1)
        line = tmp_graph.readline()
    # adj_matrix = sparse.coo_matrix((data, (row_ind, col_ind)), shape=[len(node_list), len(node_list)])
    adj_matrix = coo_matrix((data, (row_ind, col_ind)), shape=[len(node_list), len(node_list)])
    return adj_matrix, node_list


def load_constraints(cons_file):
    f = open(cons_file, "r", encoding='utf-8').readlines()
    constraints = [int(a.replace("\n", "")) for a in f]
    constraints = np.array(constraints)
    return constraints


def adj_to_triple(adj):
    """
    :param adj: adjacent matrix
    :return: triple set -- numpy array -- [row_index, col_index, edge_state]
    """
    # if type(adj) is sparse.coo_matrix:
    #     adjacent_matrix = adj.tocsr()
    if isinstance(adj, coo_matrix):
        adjacent_matrix = adj.tocsr()

    node_number = adjacent_matrix.shape[0]
    triple = []
    for zi in range(node_number):
        # triple.append([zi, zi, adjacent_matrix[zi, zi]])
        for zj in range(zi + 1, node_number):
            triple.append([zi, zj, adjacent_matrix[zi, zj]])
            triple.append([zj, zi, adjacent_matrix[zj, zi]])
    return np.array(triple)


def get_subset_of_training_set(test_feature, X_train, m):
    test_feature_tmp = np.array(test_feature)[np.newaxis, :]
    axis = tuple(np.arange(1, X_train.ndim, dtype=np.int32))
    dist = np.sum(np.square(X_train - test_feature_tmp), axis=axis)
    dist = np.squeeze(dist)
    ind = np.argsort(dist)
    ind = np.squeeze(ind).transpose()
    return ind[:m]
