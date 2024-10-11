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
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

package_class = {}

def extract_degree_centrality(sensitive_apis, original_file, output_file, label, year):
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

def extract_degree_centrality_gexfs(sensitive_apis, gexfs, output_file, label, year):
    Vectors = []
    Labels = []

    if gexfs is not None and len(gexfs) > 1:
        pool = ThreadPool(50)
        vector = pool.map(partial(degree_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        Vectors.extend(vector)
        Labels.extend([label for i in range(len(vector))])
    elif gexfs is not None and len(gexfs) == 1:
        vector = degree_centrality_feature(gexfs, sensitive_apis)
        Vectors.append(vector)
        Labels.append(label)

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
        sha256 = None
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 is not None and sha256 not in existing_sha256:  # check if sha256 already exists
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
    Vectors = []
    Labels = []

    if os.path.isdir(original_file):
        if original_file[-1] == '/':
            gexfs = glob.glob(original_file + '*.gexf')
        else:
            gexfs = glob.glob(original_file + '/*.gexf')

        for file in gexfs:
            vector = katz_centrality_feature(file, sensitive_apis)
            Vectors.append(vector)
            Labels.append(label)

    else:
        vector = katz_centrality_feature(original_file, sensitive_apis)
        Vectors.append(vector)
        Labels.append(label)

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

def extract_katz_centrality_gexfs(sensitive_apis, gexfs, output_file, label, year):

    Vectors = []
    Labels = []

    if gexfs is not None and len(gexfs) > 1:
        for file in gexfs:
            try:
                vector = katz_centrality_feature(file, sensitive_apis)
                if vector is not None:
                    Vectors.append(vector)
                    Labels.append(label)
            except Exception as e:
                print("Error parsing file:{},e:{}".format(file,e))
    elif gexfs is not None and len(gexfs) == 1:
        vector = katz_centrality_feature(gexfs, sensitive_apis)
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
        sha256 = None
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 is not None and sha256 not in existing_sha256:  # check if sha256 already exists
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
    Vectors = []
    Labels = []

    if os.path.isdir(orignal_file):
        if orignal_file[-1] == '/':
            gexfs = glob.glob(orignal_file + '*.gexf')
        else:
            gexfs = glob.glob(orignal_file + '/*.gexf')

        pool = ThreadPool(50)
        vector = pool.map(partial(harmonic_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        Vectors.extend(vector)
        Labels.extend([label for i in range(len(vector))])
    else:
        vector = harmonic_centrality_feature(orignal_file, sensitive_apis)
        Vectors.append(vector)
        Labels.append(label)

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
        sha256 = None
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 is not None and sha256 not in existing_sha256:  # check if sha256 already exists
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

def extract_harmonic_centrality_gexfs(sensitive_apis, gexfs, output_file, label, year):
    Vectors = []
    Labels = []

    if gexfs is not None and len(gexfs) > 1:
        pool = ThreadPool(50)
        vector = pool.map(partial(harmonic_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        Vectors.extend(vector)
        Labels.extend([label for i in range(len(vector))])
    elif gexfs is not None and len(gexfs) == 1:
        vector = harmonic_centrality_feature(sensitive_apis, gexfs)
        Vectors.append(vector)
        Labels.append(label)

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
        sha256 = None
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 is not None and sha256 not in existing_sha256:  # check if sha256 already exists
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
    Vectors = []
    Labels = []

    if os.path.isdir(orignal_file):
        if orignal_file[-1] == '/':
            gexfs = glob.glob(orignal_file + '*.gexf')
        else:
            gexfs = glob.glob(orignal_file + '/*.gexf')

        pool = ThreadPool(50)
        vector = pool.map(partial(closeness_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        Vectors.extend(vector)
        Labels.extend([label for i in range(len(vector))])
    else:
        vector = closeness_centrality_feature(orignal_file, sensitive_apis)
        Vectors.append(vector)
        Labels.append(label)

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
        sha256 = None
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 is not None and sha256 not in existing_sha256:  # check if sha256 already exists
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

def extract_closeness_centrality_gexfs(sensitive_apis, gexfs, output_file, label, year):
    Vectors = []
    Labels = []

    if gexfs is not None and len(gexfs) > 1:

        pool = ThreadPool(50)
        vector = pool.map(partial(closeness_centrality_feature, sensitive_apis=sensitive_apis), gexfs)
        Vectors.extend(vector)
        Labels.extend([label for i in range(len(vector))])
    else:
        vector = closeness_centrality_feature(sensitive_apis, gexfs)
        Vectors.append(vector)
        Labels.append(label)

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
        sha256 = None
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        if sha256 is not None and sha256 not in existing_sha256:  # check if sha256 already exists
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
    file = "sensitive_apis.txt"
    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            else:
                sensitive_apis.append(line.strip())
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
    sha256 = file.split('/')[-1].split('.')[0]
    try:
        CG = callgraph_extraction(file)
        # First attempt: using default parameters
        try:
            node_centrality = nx.katz_centrality(CG)
        except nx.PowerIterationFailedConvergence:
            # Second attempt: using custom parameters max_iter=2000, alpha=0.05
            try:
                node_centrality = nx.katz_centrality(CG, max_iter=2000, alpha=0.05)
            except nx.PowerIterationFailedConvergence:
                # Third attempt: using custom parameters max_iter=3000, alpha=0.025
                node_centrality = nx.katz_centrality(CG, max_iter=3000, alpha=0.025)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        return (sha256, vector)

    except nx.PowerIterationFailedConvergence or ET.ParseError as e:
        # Handle all exceptions and log the failed file
        print(f"Error processing file: {file}, error: {e}")
        failed_record = 'failed_record.txt'
        with open(failed_record, 'a') as f:
            f.write(file + '\n')

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

        return (sha256, vector)

    except ET.ParseError as e:
            print("Error parsing file:{},e:{}".format(file,e))
            return None


def extract_degree_centrality_from_CG(callgraph, sensitive_apis):
    try:
        N = len(callgraph)
        vector = []

        for api in sensitive_apis:
            if callgraph.has_node(api):
                centrality = callgraph.degree(api) / (N - 1)
                vector.append(centrality)
            else:
                vector.append(0)

        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph, e))
        return None


def extract_katz_centrality_from_CG(callgraph, sensitive_apis):
    try:
        # First attempt: using default parameters
        try:
            node_centrality = nx.katz_centrality(callgraph)
        except nx.PowerIterationFailedConvergence:
            # Second attempt: using custom parameters max_iter=2000, alpha=0.05
            try:
                node_centrality = nx.katz_centrality(callgraph, max_iter=2000, alpha=0.05)
            except nx.PowerIterationFailedConvergence:
                # Third attempt: using custom parameters max_iter=3000, alpha=0.025
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
        # If all attempts fail, print the error and return None
        print(f"All attempts failed for file: ", e)

        return None

def extract_katz_centrality_from_CG_new(callgraph, sensitive_apis):
    try:
        n = len(callgraph.nodes())
        A = nx.to_numpy_array(callgraph)  # Convert the graph to an adjacency matrix
        alpha = 0.025  # Can be adjusted as needed
        beta = 1.0  # May need to be adjusted based on requirements

        # Construct the matrix and vector for Katz centrality formula
        I = np.eye(n)  # Create an identity matrix
        b = np.ones(n) * beta
        A_katz = np.linalg.inv(I - alpha * A.T) - I  # Compute (I - alpha*A^T)^-1 - I
        katz_scores = A_katz.dot(b)

        # Map the results back to the nodes
        node_centrality = {list(callgraph.nodes())[i]: score for i, score in enumerate(katz_scores)}

        # Construct the feature vector
        vector = [node_centrality.get(api, 0) for api in sensitive_apis]

        return vector
    except Exception as e:
        print(f"Error processing callgraph:{callgraph}, error:{e}")
        return None



def extract_harmonic_centrality_from_CG(callgraph, sensitive_apis):
    try:

        node_centrality = nx.harmonic_centrality(callgraph)

        vector = []
        for api in sensitive_apis:
            if api in node_centrality.keys():
                vector.append(node_centrality[api])
            else:
                vector.append(0)

        if not check_vector(vector):
            return None

        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph,e))
        return None

def extract_harmonic_centrality_from_CG_new(callgraph, sensitive_apis, sensitive_api_idx):
    try:
        node_centrality = nx.harmonic_centrality(callgraph, nbunch=sensitive_api_idx)

        vector = [node_centrality.get(api, 0) for api in sensitive_apis]

        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph,e))
        return None

def extract_closeness_centrality_from_CG(callgraph, sensitive_apis):
    try:
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
        reversed_callgraph = callgraph.reverse(copy=True)  # Create a reversed graph to account for in-degree and out-degree in directed graphs

        vector = []
        for node in sensitive_apis:
            if node != -1:
                sp = nx.single_source_shortest_path(reversed_callgraph, node)
                totsp = sum([len(path) for path in sp.values()])  # Calculate the total distance
                len_G = len(reversed_callgraph)  # Get the total number of nodes in the graph
                _closeness_centrality = 0.0  # Initialize closeness centrality value to 0.0

                if totsp > 0.0 and len_G > 1:
                    _closeness_centrality = (len(sp) - 1.0) / totsp

                    # Further normalize closeness centrality based on the wf_improved option
                    s = (len(sp) - 1.0) / (len_G - 1)
                    _closeness_centrality *= s

                vector.append(_closeness_centrality)
            else:
                vector.append(0)

        return vector
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(callgraph, e))
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


def smail_to_abstract(func_sin, abstract_list):
    class_name = func_sin.split(";")[0][1:].replace("/", ".")

    for abstract in abstract_list:
        if class_name.startswith(abstract):
            return abstract

    items = class_name.split('.')
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
            return package

    item_len = len(items)
    count_l = 0
    for item in items:
        if len(item) < 3:
            count_l += 1
    if count_l > (item_len / 2):
        return "obfuscated"
    else:
        return "self-defined"


def mamadroid_feature(file):
    sha256 = file.split('/')[-1].split('.')[0]

    try:

        CG = callgraph_extraction(file)
        families = []
        with open('../type/families.txt', "r") as f:
            for line in f:
                families.append(line.strip())

        packages = []
        with open('../type/packages.txt', "r") as f:
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
    with open('../type/families.txt', "r") as f:
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
        with open('../type/families.txt', "r") as f:
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


def apigraph_features(file):
    sha256 = file.split('/')[-1].split('.')[0]
    try:
        CG = callgraph_extraction(file)
        dim =50
        apigraph_map_file = "../type/entity_class{}.txt".format(dim)
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
        return sha256, markov_apigraph_features
    except ET.ParseError as e:
        print("Error parsing file:{},e:{}".format(file, e))
        return None


def extract_apigraph_features_gexfs(gexfs, output_file, label, year):
    Vectors = []
    Labels = []

    if gexfs is not None and len(gexfs) > 1:
        pool = ThreadPool(50)
        vector = pool.map(partial(apigraph_features), gexfs)
        Vectors.extend(vector)
        Labels.extend([label for i in range(len(vector))])
    elif len(gexfs) == 1:
        vector = apigraph_features(gexfs)
        Vectors.append(vector)
        Labels.append(label)


    if not os.path.exists(output_file):
        os.mkdir(output_file)

    if output_file[-1] == '/':
        csv_path = output_file + "apigraph_" + str(label) + "_" + year + '.csv'
    else:
        csv_path = output_file + "/apigraph_" + str(label) + "_" + year + '.csv'

    rows_to_write = []
    for i in range(len(Labels)):
        vector_item = Vectors[i]
        if vector_item is not None and len(vector_item) == 2:
            (sha256, vector) = vector_item
        # if sha256 not in existing_sha256:  # check if sha256 already exists
            row = [sha256]
            row.extend(vector)
            row.append(Labels[i])
            rows_to_write.append(row)

    # Write the rows to the CSV file
    if rows_to_write:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)




def extract_mamadroid_family_features_gexfs(gexfs, output_file, label, year):

    Family_Vectors = []
    Sha256 = []
    Labels = []

    if gexfs is not None and len(gexfs) > 1:


        pool = ThreadPool(20)
        results = pool.map(mamadroid_family_feature, gexfs)
        for vector in results:
            if vector is not None:
                sha256, family_vector = vector
                Sha256.append(sha256)
                Family_Vectors.append(family_vector)
                Labels.append(label)
    elif len(gexfs) == 1:
        vector = mamadroid_family_feature(gexfs)
        if vector is not None and len(vector) == 2:
            sha256, family_vector = vector
            Sha256.append(sha256)
            Family_Vectors.append(family_vector)
            Labels.append(label)


    if not os.path.exists(output_file):
        os.mkdir(output_file)

    Family_Vectors = csr_matrix(Family_Vectors)


    data_to_dava_family = (Sha256, Family_Vectors, Labels)


    if not output_file.endswith('/'):
        output_file = output_file + '/'
    with open(f"{output_file}mamadroid_family_{label}_{year}.pkl", 'wb') as f:
        pickle.dump(data_to_dava_family, f)

    print("Data extraction and saving completed.")


def normalizing_matrix(call_matrix):
    row_sums = call_matrix.sum(axis=1)
    row_sums = row_sums[:, np.newaxis]
    normalized_matrix = call_matrix / row_sums
    normalized_matrix = np.nan_to_num(normalized_matrix)

    return normalized_matrix

