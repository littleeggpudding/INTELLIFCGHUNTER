import networkx as nx
import time
import argparse
import csv
from multiprocessing import Pool as ThreadPool
from functools import partial
import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import pickle
import pandas as pd
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

package_class = {}

def extract_degree_centrality(sensitive_apis, original_file, output_file, label, year):
    #提取数据
    Vectors = []
    Labels = []

    if os.path.isdir(original_file):
        if original_file[-1] == '/':
            gexfs = glob.glob(original_file + '*.gexf')
        else:
            gexfs = glob.glob(original_file + '/*.gexf')

        gexfs = gexfs[:5000]

        print("gexfs", len(gexfs))



        pool = ThreadPool(50)
        vector = pool.map(partial(degree_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        Vectors.extend(vector)
        Labels.extend([label for i in range(len(vector))])
    else:
        vector = degree_centrality_feature(original_file, sensitive_apis)
        Vectors.append(vector)
        Labels.append(label)

    print("finish!!!!")
    print("dataset", len(vector))

    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    if output_file[-1] == '/':
        csv_path = output_file + "degree_" + str(label) + "_" + year + '.csv'
    else:
        csv_path = output_file + "/degree_" + str(label) + "_" + year + '.csv'

    # Read existing sha256 values from the CSV file
    existing_sha256 = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                existing_sha256.add(row[0])

    # Prepare the rows to be written
    rows_to_write = []
    for i in range(len(Labels)):
        vector_item = Vectors[i]
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 not in existing_sha256:  # check if sha256 already exists
            row = [sha256]
            row.extend(vector)
            row.append(Labels[i])
            rows_to_write.append(row)

    # Write the rows to the CSV file
    if rows_to_write:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Check if file is empty (no header yet)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(['SHA256'] + sensitive_apis + ['Label'])
            writer.writerows(rows_to_write)

def obtain_sha256(feature_file, label, year):
    sha256 = []
    sha256_visited = set()
    target_file = feature_file + "degree_" + str(label) + "_" + year + '.csv'
    with open(target_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] not in sha256_visited:
                sha256.append(row[0])
                sha256_visited.add(row[0])

            if len(sha256) >= 1000:
                break
    return sha256

def extract_katz_centrality_1000samples(sensitive_apis, original_file, output_file, label, year):
    #提取数据
    Vectors = []
    Labels = []

    sha256_list = obtain_sha256(output_file, label, year)
    if os.path.isdir(original_file):
        if original_file[-1] == '/':
            gexfs = [original_file + sha256 + '.gexf' for sha256 in sha256_list]
        else:
            gexfs = [original_file + '/' + sha256 + '.gexf' for sha256 in sha256_list]

        print("gexfs", len(gexfs))

        # pool = ThreadPool(5)
        # vector = pool.map(partial(katz_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        for file in gexfs:
            vector = katz_centrality_feature(file, sensitive_apis)
            Vectors.append(vector)
            Labels.append(label)
            print("finish one")
        # Vectors.extend(vector)
        # Labels.extend([label for i in range(len(vector))])
    else:
        vector = katz_centrality_feature(original_file, sensitive_apis)
        Vectors.append(vector)
        Labels.append(label)

    print("finish!!!!")
    print("dataset", len(Vectors))

    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    if output_file[-1] == '/':
        csv_path = output_file + "new_katz_" + str(label) + "_" + year + '.csv'
    else:
        csv_path = output_file + "/new_katz_" + str(label) + "_" + year + '.csv'


    # Prepare the rows to be written
    rows_to_write = []
    for i in range(len(Labels)):
        vector_item = Vectors[i]
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
            row = [sha256]
            row.extend(vector)
            row.append(Labels[i])
            rows_to_write.append(row)

    # Write the rows to the CSV file
    if rows_to_write:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Check if file is empty (no header yet)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(['SHA256'] + sensitive_apis + ['Label'])
            writer.writerows(rows_to_write)

def extract_katz_centrality(sensitive_apis, original_file, output_file, label, year):
    #提取数据
    Vectors = []
    Labels = []

    if os.path.isdir(original_file):
        if original_file[-1] == '/':
            gexfs = glob.glob(original_file + '*.gexf')
        else:
            gexfs = glob.glob(original_file + '/*.gexf')

        gexfs = gexfs[:5000]
        print("gexfs", len(gexfs))

        # pool = ThreadPool(5)
        # vector = pool.map(partial(katz_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        for file in gexfs:
            vector = katz_centrality_feature(file, sensitive_apis)
            Vectors.append(vector)
            Labels.append(label)
            print("finish one")
        # Vectors.extend(vector)
        # Labels.extend([label for i in range(len(vector))])
    else:
        vector = katz_centrality_feature(original_file, sensitive_apis)
        Vectors.append(vector)
        Labels.append(label)

    print("finish!!!!")
    print("dataset", len(Vectors))

    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    if output_file[-1] == '/':
        csv_path = output_file + "katz_" + str(label) + "_" + year + '.csv'
    else:
        csv_path = output_file + "/katz_" + str(label) + "_" + year + '.csv'

    # Read existing sha256 values from the CSV file
    existing_sha256 = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                existing_sha256.add(row[0])

    # Prepare the rows to be written
    rows_to_write = []
    for i in range(len(Labels)):
        vector_item = Vectors[i]
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 not in existing_sha256:  # check if sha256 already exists
            row = [sha256]
            row.extend(vector)
            row.append(Labels[i])
            rows_to_write.append(row)

    # Write the rows to the CSV file
    if rows_to_write:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Check if file is empty (no header yet)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(['SHA256'] + sensitive_apis + ['Label'])
            writer.writerows(rows_to_write)

def extract_harmonic_centrality(sensitive_apis, orignal_file, output_file, label, year):
    #提取数据
    Vectors = []
    Labels = []

    if os.path.isdir(orignal_file):
        if orignal_file[-1] == '/':
            gexfs = glob.glob(orignal_file + '*.gexf')
        else:
            gexfs = glob.glob(orignal_file + '/*.gexf')

        gexfs = gexfs[:5000]

        print("gexfs", len(gexfs))

        pool = ThreadPool(50)
        vector = pool.map(partial(harmonic_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        Vectors.extend(vector)
        Labels.extend([label for i in range(len(vector))])
    else:
        vector = harmonic_centrality_feature(orignal_file, sensitive_apis)
        Vectors.append(vector)
        Labels.append(label)

    print("finish!!!!")
    print("dataset", len(vector))

    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    if output_file[-1] == '/':
        csv_path = output_file + "harmonic_" + str(label) + "_" + year + '.csv'
    else:
        csv_path = output_file + "/harmonic_" + str(label) + "_" + year + '.csv'

    # Read existing sha256 values from the CSV file
    existing_sha256 = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                existing_sha256.add(row[0])

    # Prepare the rows to be written
    rows_to_write = []
    for i in range(len(Labels)):
        vector_item = Vectors[i]
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 not in existing_sha256:  # check if sha256 already exists
            row = [sha256]
            row.extend(vector)
            row.append(Labels[i])
            rows_to_write.append(row)

    # Write the rows to the CSV file
    if rows_to_write:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Check if file is empty (no header yet)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(['SHA256'] + sensitive_apis + ['Label'])
            writer.writerows(rows_to_write)

def extract_closeness_centrality(sensitive_apis, orignal_file, output_file, label, year):
    #提取数据
    Vectors = []
    Labels = []

    if os.path.isdir(orignal_file):
        if orignal_file[-1] == '/':
            gexfs = glob.glob(orignal_file + '*.gexf')
        else:
            gexfs = glob.glob(orignal_file + '/*.gexf')

        gexfs = gexfs[:5000]
        print("gexfs", len(gexfs))

        pool = ThreadPool(50)
        vector = pool.map(partial(closeness_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        Vectors.extend(vector)
        Labels.extend([label for i in range(len(vector))])
    else:
        vector = closeness_centrality_feature(orignal_file, sensitive_apis)
        Vectors.append(vector)
        Labels.append(label)

    print("finish!!!!")
    print("dataset", len(vector))

    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    if output_file[-1] == '/':
        csv_path = output_file + "closeness_" + str(label) + "_" + year + '.csv'
    else:
        csv_path = output_file + "/closeness_" + str(label) + "_" + year + '.csv'

    # Read existing sha256 values from the CSV file
    existing_sha256 = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                existing_sha256.add(row[0])

    # Prepare the rows to be written
    rows_to_write = []
    for i in range(len(Labels)):
        vector_item = Vectors[i]
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 not in existing_sha256:  # check if sha256 already exists
            row = [sha256]
            row.extend(vector)
            row.append(Labels[i])
            rows_to_write.append(row)

    # Write the rows to the CSV file
    if rows_to_write:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Check if file is empty (no header yet)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(['SHA256'] + sensitive_apis + ['Label'])
            writer.writerows(rows_to_write)


def obtain_sensitive_apis():
    # file = 'sensitive_apis.txt'
    file = "important_sensitive_apis.txt"
    # file = 'new_important_sensitive_apis_based_on_dt.txt'
    print(file)
    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            else:
                sensitive_apis.append(line.strip())
    #close


    return sensitive_apis


def callgraph_extraction(file):
    CG = nx.read_gexf(file)
    return CG

def degree_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.')[0]
    try:
        CG = callgraph_extraction(file)
        # print(file)
        node_centrality = nx.degree_centrality(CG)

        # print(node_centrality)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        if not check_vector(vector):
            return None
        # print(file)
        return (sha256, vector)
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(file,e))
        return None

def check_vector(vector):
    for i in range(len(vector)):
        if vector[i] != 0.0:
            return True
    return False

def katz_centrality_feature(file, sensitive_apis):
    # sha256 = file.split('/')[-1].split('.')[0]
    # try:
    #     CG = callgraph_extraction(file)
    #     node_centrality = nx.katz_centrality(CG)
    #     # node_centrality = nx.katz_centrality(CG, max_iter=2000, alpha=0.025)
    #
    #     vector = []
    #     for api in sensitive_apis:
    #         if api in node_centrality.keys():
    #             vector.append(node_centrality[api])
    #         else:
    #             vector.append(0)
    #         # print(file)
    #     if not check_vector(vector):
    #         return None
    #
    #     return (sha256, vector)
    # except nx.PowerIterationFailedConvergence as e:
    #     print("Error parsing file:{},e:{}".format(file,e))
    #     return None
    sha256 = file.split('/')[-1].split('.')[0]
    try:
        CG = callgraph_extraction(file)
        # 第一次尝试：使用默认参数
        try:
            node_centrality = nx.katz_centrality(CG)
        except nx.PowerIterationFailedConvergence:
            # 第二次尝试：使用自定义参数 max_iter=2000, alpha=0.05
            try:
                node_centrality = nx.katz_centrality(CG, max_iter=2000, alpha=0.05)
            except nx.PowerIterationFailedConvergence:
                # 第三次尝试：使用自定义参数 max_iter=3000, alpha=0.025
                node_centrality = nx.katz_centrality(CG, max_iter=3000, alpha=0.025)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        if not check_vector(vector):
            return None

        return (sha256, vector)

    except nx.PowerIterationFailedConvergence as e:
        # 如果所有尝试都失败了，打印错误并返回 None
        print(f"All attempts failed for file: {file}, error: {e}")
        failed_record = 'failed_record.txt'
        with open(failed_record, 'a') as f:
            f.write(file + '\n')

        return None


def katz_centrality_feature_pytorch(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.')[0]
    try:
        CG = callgraph_extraction(file)
        n = len(CG.nodes())
        A = nx.to_numpy_array(CG)  # 将图转换为邻接数组
        A = torch.from_numpy(A).float()  # 转换为 PyTorch 张量

        alpha = 0.025  # 可以根据需要调整
        beta = 1.0  # 根据你的需求可能需要调整

        # 使用 PyTorch 进行 Katz centrality 计算
        I = torch.eye(n)  # 创建单位矩阵
        b = torch.ones(n, 1) * beta
        A_katz = torch.linalg.inv(I - alpha * A.t()) - I  # 计算 (I - alpha*A^T)^-1 - I
        katz_scores = A_katz.matmul(b).squeeze()

        # 将计算结果映射回节点
        node_centrality = {list(CG.nodes())[i]: katz_scores[i].item() for i in range(n)}

        # 构建特征向量
        vector = [node_centrality.get(api, 0) for api in sensitive_apis]

        # 检查向量
        if not check_vector(vector):
            return None

        return (sha256, vector)
    except ET.ParseError as e:
        print(f"Error parsing file: {file}, error: {e}")
        return None

def closeness_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.')[0]
    try:
        CG = callgraph_extraction(file)
        node_centrality = nx.closeness_centrality(CG)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        if not check_vector(vector):
            return None

        return (sha256, vector)

    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(file,e))
        return None

def harmonic_centrality_feature(file, sensitive_apis):
    sha256 = file.split('/')[-1].split('.')[0]
    try:
        CG = callgraph_extraction(file)
        node_centrality = nx.harmonic_centrality(CG)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        if not check_vector(vector):
            return None

        return (sha256, vector)

    except ET.ParseError as e:
            print("Error parsing file:{},e:{}".format(file,e))
            return None

def extract_degree_centrality_from_CG_old(callgraph, sensitive_apis):
    try:
        # print(callgraph)
        node_centrality = nx.degree_centrality(callgraph)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        if not check_vector(vector):
            return None
        #   只需要返回一个特征向量
        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph,e))
        return None

def extract_degree_centrality_from_CG(callgraph, sensitive_apis):
    try:
        N = len(callgraph)  # 获取图中的节点总数
        vector = []

        for api in sensitive_apis:
            if callgraph.has_node(api):
                # 计算节点的度中心性，即节点的度数除以 (N-1)
                centrality = callgraph.degree(api) / (N - 1)
                vector.append(centrality)
            else:
                vector.append(0)

        # if not check_vector(vector):
        #     return None

        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph, e))
        return None


def extract_katz_centrality_from_CG(callgraph, sensitive_apis):
    # try:
    #     # print(file)
    #     node_centrality = nx.katz_centrality(callgraph, max_iter=3000, alpha=0.025)
    #
    #     # vector = []
    #     # for api in sensitive_apis:
    #     #     if api in node_centrality.keys():
    #     #         vector.append(node_centrality[api])
    #     #     else:
    #     #         vector.append(0)
    #     vector = [node_centrality.get(api, 0) for api in sensitive_apis]
    #
    #     # if not check_vector(vector):
    #     #     return None
    #     #   只需要返回一个特征向量
    #     return vector
    # except ET.ParseError as e:
    #     print("Error parsing file:{},e:{}".format(callgraph,e))
    #     return None

    try:
        # 第一次尝试：使用默认参数
        try:
            node_centrality = nx.katz_centrality(callgraph)
        except nx.PowerIterationFailedConvergence:
            # 第二次尝试：使用自定义参数 max_iter=2000, alpha=0.05
            try:
                node_centrality = nx.katz_centrality(callgraph, max_iter=2000, alpha=0.05)
            except nx.PowerIterationFailedConvergence:
                # 第三次尝试：使用自定义参数 max_iter=3000, alpha=0.025
                node_centrality = nx.katz_centrality(callgraph, max_iter=3000, alpha=0.025)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        if not check_vector(vector):
            return None

        return vector

    except nx.PowerIterationFailedConvergence as e:
        # 如果所有尝试都失败了，打印错误并返回 None
        print(f"All attempts failed for file: ", e)

        return None


def extract_katz_centrality_from_CG_new_pytorch(callgraph, sensitive_apis, device='cuda'):
    try:
        n = len(callgraph.nodes())
        A_np = nx.to_numpy_array(callgraph)  # 将图转换为邻接矩阵
        A = torch.tensor(A_np, dtype=torch.float32, device=device)  # 转换为 PyTorch tensor

        alpha = 0.025  # 可以根据需要调整
        beta = 1.0  # 根据你的需求可能需要调整

        # 构建 Katz centrality 公式中的矩阵和向量
        I = torch.eye(n, device=device)  # 创建单位矩阵
        b = torch.ones(n, device=device) * beta
        A_katz = torch.linalg.inv(I - alpha * A.T) - I  # 计算 (I - alpha*A^T)^-1 - I
        katz_scores = A_katz @ b

        # 将计算结果映射回节点
        node_centrality = {list(callgraph.nodes())[i]: score.item() for i, score in enumerate(katz_scores)}

        # 构建特征向量
        vector = [node_centrality.get(api, 0) for api in sensitive_apis]

        return vector
    except Exception as e:
        print(f"Error processing callgraph:{callgraph}, error:{e}")
        return None

def extract_katz_centrality_from_CG_new(callgraph, sensitive_apis):
    try:
        n = len(callgraph.nodes())
        A = nx.to_numpy_array(callgraph)  # 将图转换为邻接矩阵
        alpha = 0.025  # 可以根据需要调整
        beta = 1.0  # 根据你的需求可能需要调整

        # 构建 Katz centrality 公式中的矩阵和向量
        I = np.eye(n)  # 创建单位矩阵
        b = np.ones(n) * beta
        A_katz = np.linalg.inv(I - alpha * A.T) - I  # 计算 (I - alpha*A^T)^-1 - I
        katz_scores = A_katz.dot(b)

        # 将计算结果映射回节点
        node_centrality = {list(callgraph.nodes())[i]: score for i, score in enumerate(katz_scores)}

        # 构建特征向量
        vector = [node_centrality.get(api, 0) for api in sensitive_apis]

        return vector
    except Exception as e:
        print(f"Error processing callgraph:{callgraph}, error:{e}")
        return None

def extract_harmonic_centrality_from_CG(callgraph, sensitive_apis):
    try:
        # print(file)
        node_centrality = nx.harmonic_centrality(callgraph)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        if not check_vector(vector):
            return None
        #   只需要返回一个特征向量
        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph,e))
        return None

def extract_harmonic_centrality_from_CG_new(callgraph, sensitive_apis, sensitive_api_idx):
    try:
        # print(file)
        node_centrality = nx.harmonic_centrality(callgraph, nbunch=sensitive_api_idx)

        vector = [node_centrality.get(api, 0) for api in sensitive_apis]

        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph,e))
        return None

def extract_closeness_centrality_from_CG(callgraph, sensitive_apis):
    try:
        # print(file)
        node_centrality = nx.closeness_centrality(callgraph)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph,e))
        return None

def extract_closeness_centrality_from_CG_new(callgraph, sensitive_apis):
    try:
        reversed_callgraph = callgraph.reverse(copy=True)  # 创建一个反向图视图，以考虑有向图的入度和出度
        # path_length = nx.single_source_shortest_path_length

        vector = []
        for node in sensitive_apis:
            if node != -1:
                # sp = path_length(reversed_callgraph, node)# 计算节点 n 到其他节点的距离
                sp = nx.single_source_shortest_path(reversed_callgraph, node)
                # 打印最短路径信息
                # for target_node, path in sp.items():
                #     print(f"Shortest path from {node} to {target_node}: {path}")
                # totsp = sum(sp.values())  # 计算总距离
                # print(f"Total distance (sum of all shortest paths) from {node} to other nodes: {totsp}")
                totsp = sum([len(path) for path in sp.values()])  # 计算总距离
                len_G = len(reversed_callgraph)  # 获取图中节点的总数
                _closeness_centrality = 0.0  # 初始化紧密度中心性值为0.0

                if totsp > 0.0 and len_G > 1:
                    _closeness_centrality = (len(sp) - 1.0) / totsp

                    # 根据 wf_improved 选项进一步标准化紧密度中心性
                    s = (len(sp) - 1.0) / (len_G - 1)
                    _closeness_centrality *= s

                vector.append(_closeness_centrality)
            else:
                vector.append(0)

        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph,e))
        return None

def extract_closeness_harmonic_centrality_from_CG(callgraph, sensitive_apis):
    try:
        reversed_callgraph = callgraph.reverse(copy=True)

        N = len(reversed_callgraph)
        harmonic_vector = []
        closeness_vector = []

        for node in sensitive_apis:
            if not reversed_callgraph.has_node(node):
                harmonic_vector.append(0)
                closeness_vector.append(0)
                continue

            path_lengths = nx.single_source_shortest_path_length(reversed_callgraph, node)
            harmonic_sum = sum(1 / distance if distance > 0 else 0 for distance in path_lengths.values())
            closeness_sum = sum(path_lengths.values())

            harmonic_centrality = harmonic_sum / (N - 1)
            closeness_centrality = 1 / closeness_sum if closeness_sum > 0 else 0

            harmonic_vector.append(harmonic_centrality)
            closeness_vector.append(closeness_centrality)

        return closeness_vector, harmonic_vector
    except Exception as e:
        print("Error:{}".format(e))
        return None, None

# def extract_degree_centrality_from_adj(adj_matrix, sensitive_api_idx):



#get number of nodes in a graph
def get_number_of_nodes(graph):
    return graph.number_of_nodes()

def extract_node_counts_from_directory(directory_path):
    counts = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.gexf'):
            print("11111")
            full_path = os.path.join(directory_path, filename)
            print("full_path: ",full_path)
            CG = callgraph_extraction(full_path)
            counts[filename] = get_number_of_nodes(CG)
    return counts


def smail_to_abstract(func_sin, abstract_list):
    # print("family func_sin", func_sin)
    class_name = func_sin.split(";")[0][1:].replace("/", ".")
    # print("family class_name", class_name)
    for abstract in abstract_list:
        if class_name.startswith(abstract):
            return abstract

    items = class_name.split('.')
    # print("family items", items)
    item_len = len(items)
    count_l = 0
    for item in items:
        if len(item) < 3:
            count_l += 1
    if count_l > (item_len / 2):
        return "obfuscated"
    else:
        return "self-defined"


def smail_to_package(func_sin, package_list):
    # print("func_sin", func_sin)
    class_name = func_sin.split(";")[0][1:].replace("/", ".")
    # print("class_name", class_name)

    items = class_name.split('.')
    packages = []
    packages.append(items[0])
    for i in range(1, len(items)):
        cur_package = packages[i - 1] + '.' + items[i]
        packages.append(cur_package)

    packages.reverse()
    # print("family packages", packages)
    for package in packages:
        if package in package_list:
            # print("cur_package", package)
            return package

    # print("family items", items)
    item_len = len(items)
    count_l = 0
    for item in items:
        if len(item) < 3:
            count_l += 1
    if count_l > (item_len / 2):
        return "obfuscated"
    else:
        return "self-defined"


def apigraph_feature(file):
    sha256 = file.split('/')[-1].split('.')[0]
    try:
        CG = callgraph_extraction(file)
        families = []
        with open('families.txt', "r") as f:
            for line in f:
                families.append(line.strip())

        packages = []
        with open('packages.txt', "r") as f:
            for line in f:
                packages.append(line.strip())

        families.append("self-defined")
        families.append("obfuscated")
        markov_family_features = np.zeros((len(families), len(families)))

        packages.append("self-defined")
        packages.append("obfuscated")
        markov_package_features = np.zeros((len(packages), len(packages)))
        assert package_class
        if CG is not None:
            total_edges = len(CG.edges())
            for edge in CG.edges():
                caller = edge[0]
                callee = edge[1]
                # caller_family = smail_to_abstract(caller, families[:-2])
                # callee_family = smail_to_abstract(callee, families[:-2])
                # caller_family_index = families.index(caller_family)
                # callee_family_index = families.index(callee_family)
                # markov_family_features[caller_family_index][callee_family_index] += 1

                caller_package = smail_to_package(caller, packages[:-2])
                callee_package = smail_to_package(callee, packages[:-2])
                caller_package_index = packages.index(caller_package)
                callee_package_index = packages.index(callee_package)
                markov_package_features[caller_package_index][callee_package_index] += 1

            if total_edges != 0:
                # print("markov_family_features", markov_family_features[:5][:5])
                # print("markov_package_features", markov_package_features[:5][:5])
                # markov_family_features = normalizing_matrix(markov_family_features)
                markov_package_features = normalizing_matrix(markov_package_features)
                # print("markov_family_features", markov_family_features[:5][:5])
                # print("markov_package_features", markov_package_features[:5][:5])
                # markov_family_features = markov_family_features.flatten()
                markov_package_features = markov_package_features.flatten()
                return sha256, markov_family_features, markov_package_features
            else:
                return None
        else:
            return None
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(file, e))
        return None
def mamadroid_feature(file):
    sha256 = file.split('/')[-1].split('.')[0]

    try:

        CG = callgraph_extraction(file)
        families = []
        with open('families.txt', "r") as f:
            for line in f:
                families.append(line.strip())

        packages = []
        with open('packages.txt', "r") as f:
            for line in f:
                packages.append(line.strip())


        families.append("self-defined")
        families.append("obfuscated")
        markov_family_features = np.zeros((len(families), len(families)))

        packages.append("self-defined")
        packages.append("obfuscated")
        markov_package_features = np.zeros((len(packages), len(packages)))


        if CG is not None:
            total_edges = len(CG.edges())
            for edge in CG.edges():
                caller = edge[0]
                callee = edge[1]
                caller_family = smail_to_abstract(caller, families[:-2])
                callee_family = smail_to_abstract(callee, families[:-2])
                caller_family_index = families.index(caller_family)
                callee_family_index = families.index(callee_family)
                markov_family_features[caller_family_index][callee_family_index] += 1

                caller_package = smail_to_package(caller, packages[:-2])
                callee_package = smail_to_package(callee, packages[:-2])
                caller_package_index = packages.index(caller_package)
                callee_package_index = packages.index(callee_package)
                markov_package_features[caller_package_index][callee_package_index] += 1


            if total_edges != 0:
                # print("markov_family_features", markov_family_features[:5][:5])
                # print("markov_package_features", markov_package_features[:5][:5])
                markov_family_features = normalizing_matrix(markov_family_features)
                markov_package_features = normalizing_matrix(markov_package_features)
                # print("markov_family_features", markov_family_features[:5][:5])
                # print("markov_package_features", markov_package_features[:5][:5])
                markov_family_features = markov_family_features.flatten()
                markov_package_features = markov_package_features.flatten()
                return sha256, markov_family_features, markov_package_features
            else:
                return None
        else:
            return None
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(file, e))
        return None
    
def extract_mamadroid_features_from_fcg(fcg):
    CG = fcg.original_call_graph
    families = []
    with open('families.txt', "r") as f:
        for line in f:
            families.append(line.strip())

    families.append("self-defined")
    families.append("obfuscated")
    detail_edges = np.zeros((len(families), len(families)))
    family_dict = {}
    for i in range(11):
        family_dict[i] = set()
    if CG is not None:
        total_edges = len(CG.edges())
        for edge in CG.edges():
            caller = edge[0]
            callee = edge[1]
            caller_family = smail_to_abstract(caller, families[:-2])
            callee_family = smail_to_abstract(callee, families[:-2])
            caller_family_index = families.index(caller_family)
            callee_family_index = families.index(callee_family)
            detail_edges[caller_family_index][callee_family_index] += 1
            family_dict[caller_family_index].add(caller)
            family_dict[callee_family_index].add(callee)
        family_count = {}
        for i in range(11):
            family_count[i] = len(family_dict[i])
        assert total_edges != 0
        markov_family_features = normalizing_matrix(detail_edges)
        markov_family_features = markov_family_features.flatten()
        return markov_family_features, detail_edges, family_count

def mamadroid_family_feature(file):
    sha256 = file.split('/')[-1].split('.')[0]

    try:

        CG = callgraph_extraction(file)
        families = []
        with open('families.txt', "r") as f:
            for line in f:
                families.append(line.strip())

        families.append("self-defined")
        families.append("obfuscated")
        markov_family_features = np.zeros((len(families), len(families)))

        if CG is not None:
            total_edges = len(CG.edges())
            for edge in CG.edges():
                caller = edge[0]
                callee = edge[1]
                caller_family = smail_to_abstract(caller, families[:-2])
                callee_family = smail_to_abstract(callee, families[:-2])
                caller_family_index = families.index(caller_family)
                callee_family_index = families.index(callee_family)
                markov_family_features[caller_family_index][callee_family_index] += 1


            if total_edges != 0:
                # print("markov_family_features", markov_family_features[:5][:5])
                # print("markov_package_features", markov_package_features[:5][:5])
                markov_family_features = normalizing_matrix(markov_family_features)
                # print("markov_family_features", markov_family_features[:5][:5])
                # print("markov_package_features", markov_package_features[:5][:5])
                markov_family_features = markov_family_features.flatten()
                return sha256, markov_family_features
            else:
                return None
        else:
            return None
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(file, e))
        return None

def obtain_malware_using_sha256(sha256_dir, gexf_dir):
    # print("sha256_dir", sha256_dir)
    # print("gexf_dir", gexf_dir)

    #读取csv
    df = pd.read_csv(sha256_dir)
    sha256_list = df.iloc[:, 0]
    gexfs = []
    for sha256 in sha256_list:
        gexf = gexf_dir + sha256 + '.gexf'
        print(gexf)
        if os.path.exists(gexf):
            gexfs.append(gexf)

    return gexfs

def extract_apigraph_features_from_fcg(fcg):
    CG = fcg.original_call_graph
    dim =50
    apigraph_map_file = "/data/b/guoqi/icse25/malwareGA/task/entity_class{}.txt".format(dim)
    package_class = {}
    with open(apigraph_map_file, 'r') as f:
        for line in f:
            package, class_num = line.strip().split(',')
            package_class[package] = int(class_num)
    packages = []
    with open('packages.txt', "r") as f:
        for line in f:
            packages.append(line.strip())

    packages.append("self-defined")
    packages.append("obfuscated")
    package_class["self-defined"] = dim
    package_class["obfuscated"] = dim + 1
    detail_edges = np.zeros((dim + 2, dim + 2))
    assert CG is not None
    assert len(CG.edges()) != 0
    apigraph_dict = {}
    for i in range(dim + 2):
        apigraph_dict[i] = set()
    for edge in CG.edges():
        caller = edge[0]
        callee = edge[1]

        caller_package = smail_to_package(caller, packages[:-2])
        callee_package = smail_to_package(callee, packages[:-2])
        # caller_package_index = packages.index(caller_package)
        # callee_package_index = packages.index(callee_package)
        # markov_package_features[caller_package_index][callee_package_index] += 1
        if caller_package in package_class and callee_package in package_class:
            caller_package_class_index = package_class[caller_package]
            callee_package_class_index = package_class[callee_package]
            detail_edges[caller_package_class_index][callee_package_class_index] += 1
            apigraph_dict[caller_package_class_index].add(caller)
            apigraph_dict[callee_package_class_index].add(callee)
    apigraph_count = {}
    for i in range(dim + 2):
        apigraph_count[i] = len(apigraph_dict[i])
    markov_apigraph_features = normalizing_matrix(detail_edges)
    markov_apigraph_features = markov_apigraph_features.flatten()
    return markov_apigraph_features, detail_edges, apigraph_count


def extract_apigraph_features(original_file, output_file, label, year):
    # 提取数据
    Family_Vectors = []
    Package_Vectors = []
    Sha256 = []
    Labels = []
    apigraph_map_file = "/data/b/shiwensong/project/malwareGA/task/entity_class30.txt"

    with open(apigraph_map_file, 'r') as f:
        for line in f:
            package, class_num = line.strip().split(',')
            package_class[package] = int(class_num)

    if os.path.isdir(original_file):
        if label == 0:
            if original_file[-1] == '/':
                gexfs = glob.glob(original_file + '*.gexf')
            else:
                gexfs = glob.glob(original_file + '/*.gexf')

        else:
            # 读取sha256
            if output_file[-1] != '/':
                output_file = output_file + '/'

            if original_file[-1] != '/':
                original_file = original_file + '/'

            dataset_file = f'{output_file}degree_1_{year}.csv'
            gexfs = obtain_malware_using_sha256(dataset_file, original_file)

        print("gexfs", len(gexfs))

        pool = ThreadPool(1)
        # results = pool.map(mamadroid_feature, gexfs)
        results = pool.map(apigraph_feature, gexfs)
        for vector in results:
            if vector is not None:
                sha256, family_vector, package_vector = vector
                Sha256.append(sha256)
                Family_Vectors.append(family_vector)
                Package_Vectors.append(package_vector)
                Labels.append(label)
    else:
        vector = mamadroid_feature(original_file)
        if vector is not None:
            sha256, family_vector, package_vector = vector
            Sha256.append(sha256)
            Family_Vectors.append(family_vector)
            Package_Vectors.append(package_vector)
            Labels.append(label)

    print("finish!!!!")
    print("dataset", len(Family_Vectors))
    exit(-1)

    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    Family_Vectors = csr_matrix(Family_Vectors)
    Package_Vectors = csr_matrix(Package_Vectors)

    print("rows_to_write_family", Family_Vectors.shape)
    print("rows_to_write_package", Package_Vectors.shape)

    data_to_dava_family = (Sha256, Family_Vectors, Labels)
    data_to_dava_package = (Sha256, Package_Vectors, Labels)

    # 保存为 .pkl 文件
    if not output_file.endswith('/'):
        output_file = output_file + '/'
    with open(f"{output_file}mamadroid_family_{label}_{year}.pkl", 'wb') as f:
        pickle.dump(data_to_dava_family, f)

    with open(f"{output_file}mamadroid_package_{label}_{year}.pkl", 'wb') as f:
        pickle.dump(data_to_dava_package, f)

    print("Data extraction and saving completed.")
def extract_mamadroid_features(original_file, output_file, label, year):
    #提取数据
    Family_Vectors = []
    Package_Vectors = []
    Sha256 = []
    Labels = []

    if os.path.isdir(original_file):
        if label == 0:
            if original_file[-1] == '/':
                gexfs = glob.glob(original_file + '*.gexf')
            else:
                gexfs = glob.glob(original_file + '/*.gexf')

        else:
            #读取sha256
            if output_file[-1] != '/':
                output_file = output_file + '/'
                
            if original_file[-1] != '/':
                original_file = original_file + '/'

            dataset_file = f'{output_file}dataset_malware_{year}.csv'
            gexfs = obtain_malware_using_sha256(dataset_file, original_file)

        print("gexfs", len(gexfs))

        pool = ThreadPool(50)
        results = pool.map(mamadroid_feature, gexfs)
        for vector in results:
            if vector is not None:
                sha256, family_vector, package_vector = vector
                Sha256.append(sha256)
                Family_Vectors.append(family_vector)
                Package_Vectors.append(package_vector)
                Labels.append(label)
    else:
        vector = mamadroid_feature(original_file)
        if vector is not None:
            sha256, family_vector, package_vector = vector
            Sha256.append(sha256)
            Family_Vectors.append(family_vector)
            Package_Vectors.append(package_vector)
            Labels.append(label)

    print("finish!!!!")
    print("dataset", len(Family_Vectors))


    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_file):
        os.mkdir(output_file)


    Family_Vectors = csr_matrix(Family_Vectors)
    Package_Vectors = csr_matrix(Package_Vectors)

    print("rows_to_write_family", Family_Vectors.shape)
    print("rows_to_write_package", Package_Vectors.shape)

    data_to_dava_family = (Sha256, Family_Vectors, Labels)
    data_to_dava_package = (Sha256, Package_Vectors, Labels)

    # 保存为 .pkl 文件
    if not output_file.endswith('/'):
        output_file = output_file + '/'
    with open( f"{output_file}mamadroid_family_{label}_{year}.pkl", 'wb') as f:
        pickle.dump(data_to_dava_family, f)

    with open( f"{output_file}mamadroid_package_{label}_{year}.pkl", 'wb') as f:
        pickle.dump(data_to_dava_package, f)


    print("Data extraction and saving completed.")

    # Write the rows to the CSV file
    # if rows_to_write_family:
    #     with open(csv_path_family, 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         # Check if file is empty (no header yet)
    #         writer.writerows(rows_to_write_family)
    #
    #     with open(csv_path_package, 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         # Check if file is empty (no header yet)
    #         writer.writerows(rows_to_write_package)


def extract_mamadroid_family_features(original_file, output_file, label, year):
    # 提取数据
    Family_Vectors = []
    Package_Vectors = []
    Sha256 = []
    Labels = []

    if os.path.isdir(original_file):
        if label == 0:
            if original_file[-1] == '/':
                gexfs = glob.glob(original_file + '*.gexf')
            else:
                gexfs = glob.glob(original_file + '/*.gexf')

        else:
            # 读取sha256
            if output_file[-1] != '/':
                output_file = output_file + '/'

            if original_file[-1] != '/':
                original_file = original_file + '/'

            dataset_file = f'{output_file}dataset_malware_{year}.csv'
            gexfs = obtain_malware_using_sha256(dataset_file, original_file)

        print("gexfs", len(gexfs))

        pool = ThreadPool(20)
        results = pool.map(mamadroid_family_feature, gexfs)
        for vector in results:
            if vector is not None:
                sha256, family_vector = vector
                Sha256.append(sha256)
                Family_Vectors.append(family_vector)
                Labels.append(label)
    else:
        vector = mamadroid_family_feature(original_file)
        if vector is not None:
            sha256, family_vector = vector
            Sha256.append(sha256)
            Family_Vectors.append(family_vector)
            Labels.append(label)

    print("finish!!!!")
    print("dataset", len(Family_Vectors))

    # 判断输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_file):
        os.mkdir(output_file)

    Family_Vectors = csr_matrix(Family_Vectors)

    print("rows_to_write_family", Family_Vectors.shape)

    data_to_dava_family = (Sha256, Family_Vectors, Labels)

    # 保存为 .pkl 文件
    if not output_file.endswith('/'):
        output_file = output_file + '/'
    with open(f"{output_file}mamadroid_family_{label}_{year}.pkl", 'wb') as f:
        pickle.dump(data_to_dava_family, f)

    print("Data extraction and saving completed.")

    # Write the rows to the CSV file
    # if rows_to_write_family:
    #     with open(csv_path_family, 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         # Check if file is empty (no header yet)
    #         writer.writerows(rows_to_write_family)
    #
    #     with open(csv_path_package, 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         # Check if file is empty (no header yet)
    #         writer.writerows(rows_to_write_package)

def normalizing_matrix(call_matrix):
    # 计算每行的总调用次数
    row_sums = call_matrix.sum(axis=1)

    # 将 row_sums 的形状从 (n,) 转换为 (n,1) 以进行广播操作
    row_sums = row_sums[:, np.newaxis]

    # 归一化矩阵
    normalized_matrix = call_matrix / row_sums

    # 替换 NaN 和 inf（可能由于除以零产生）
    normalized_matrix = np.nan_to_num(normalized_matrix)
    
    return normalized_matrix

def analyze_feature(vector):
    print('shape', vector.shape)
    # 计算非零值的数量
    nonzero_count = np.count_nonzero(vector)
    cnt = 0
    for i in range(len(vector)):
        if vector[i] == 1:
            cnt += 1
            print(i, vector[i])
    print('cnt', cnt)
    # 计算最大值
    max_value = np.max(vector)
    # 计算最小值（在非零值中计算）
    min_value_nonzero = np.min(vector[np.nonzero(vector)])
    print('nonzero_count', nonzero_count)
    print('max_value', max_value)
    print('min_value_nonzero', min_value_nonzero)


if __name__ == '__main__':
    sensitive_apis = obtain_sensitive_apis()
    # print(len(sensitive_apis))
    # gexf_path = '/data/c/shiwensong/Malscan/MalScan-code/benign_2022_gexf/'
    # feature_path = '/data/c/shiwensong/Malscan/MalScan-code/feature_Nov30/'

    gexf_path = '/data/b/shiwensong/dataset/'
    feature_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    label = 0
    year = '2023'
    print("year", year)
    # extract_degree_centrality(sensitive_apis, gexf_path,
    #                           feature_path, label, year)
    # extract_katz_centrality_1000samples(sensitive_apis, gexf_path,
    #                             feature_path, label, year)
    # sha256_list = obtain_sha256(feature_path, label, year)
    # #读取一个csv文件
    # df = pd.read_csv(f'/data/b/shiwensong/dataset/feature_Nov30/new_katz_{label}_{year}.csv')
    # print(df.shape)
    # #获取他的第一列
    # new_sha256_list = df.iloc[:, 0]
    # for i in range(len(sha256_list)):
    #     if sha256_list[i] != new_sha256_list[i]:
    #         print(sha256_list[i], new_sha256_list[i])


    # extract_harmonic_centrality(sensitive_apis, gexf_path,
    #                             feature_path, label, year)
    # extract_closeness_centrality(sensitive_apis, gexf_path,
    #                             feature_path, label, year)
    # callgraph = callgraph_extraction('/data/c/shiwensong/Malscan/MalScan-code/mutation_dataset/VirusShare_c5d6324e6330851e6ae5db5f0a7cd57c.g/exf')
    # print(extract_degree_centrality_from_CG(callgraph))
    # extract_mamadroid_family_features(gexf_path, feature_path, label, year)
    # extract_apigraph_features(gexf_path, feature_path, label, year)

    # trainset = '/data/b/shiwensong/dataset/feature_Nov30/mamadroid_family_0_2023.pkl'
    #
    # extract_apigraph_features_from_fcg



    # features1, features2 = mamadroid_feature(CG, 'test3.npz')
    # analyze_feature(features1)
    # analyze_feature(features2)

