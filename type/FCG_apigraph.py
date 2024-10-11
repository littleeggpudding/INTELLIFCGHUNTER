import sys
import os
import time
import random
import networkx as nx
import numpy as np
from collections import deque
from xml.etree import ElementTree as ET
from Mutation import Mutation

class FCG_apigraph:
    # Class handling Function Call Graph (FCG) and mutations applied to it

    number = 0  # Unique ID for new nodes to avoid conflicts
    obfuscated_number = 0
    self_defined_number = 0

    def __init__(self, file_path, label, shap_value=None):
        self.original_call_graph = None  # Graph initialized once
        self.current_call_graph = None  # Current state of the graph
        self.apk_name = None  # APK name initialized once

        # Various node types and attributes
        self.nodes = None
        self.edges = None
        self.user_defined_nodes = None
        self.system_nodes = None
        self.self_defined_nodes = None
        self.obfuscated_nodes = None

        # Label and SHAP values
        self.original_label = None
        self.shap_value = shap_value

        #  feature handling
        self.dim = 50
        self.apigraph_feature_dict = {i: set() for i in range(self.dim + 2)}
        self.apigraph_matrix = np.zeros((self.dim + 2, self.dim + 2))

        self._load(file_path, label)
        self._init_call_graph()

    def _init_call_graph(self):
        self.original_call_graph.copy()

        nodes = list(self.original_call_graph.nodes._nodes.keys())
        nodes_ids = set()
        user_defined_nodes = set()
        system_nodes = set()
        self_defined_nodes = set()
        obfuscated_nodes = set()
        families = list(self.apigraph_feature_dict.keys())

        for i in range(len(nodes)):
            nodes_ids.add(i)
            current_node = nodes[i]
            current_family = self.smail_to_abstract(current_node, families)

            if current_family == self.dim:
                user_defined_nodes.add(i)
                self_defined_nodes.add(i)
            elif current_family == self.dim + 1:
                user_defined_nodes.add(i)
                obfuscated_nodes.add(i)
            else:
                system_nodes.add(i)

            self.apigraph_feature_dict[current_family].add(i)

        node_to_id = {node: idx for idx, node in enumerate(nodes)}

        self.nodes = nodes_ids
        self.system_nodes = system_nodes
        self.user_defined_nodes = user_defined_nodes
        self.self_defined_nodes = self_defined_nodes
        self.obfuscated_nodes = obfuscated_nodes

        edges = list(self.original_call_graph.edges)
        edges_ids = set()

        for edge in edges:
            u, v = edge
            u_id = node_to_id[u]
            v_id = node_to_id[v]
            edge_id = (u_id, v_id)
            edges_ids.add(edge_id)
            u_family = self.smail_to_abstract(u, families)
            v_family = self.smail_to_abstract(v, families)

            u_family_id = u_family
            v_family_id = v_family

            self.apigraph_matrix[u_family_id][v_family_id] += 1

        self.edges = edges_ids
        self.generate_new_call_graph()


    def normalizing_matrix(self):
        row_sums = self.apigraph_matrix.sum(axis=1)
        row_sums = row_sums[:, np.newaxis]

        normalized_matrix = self.apigraph_matrix / row_sums
        normalized_matrix = np.nan_to_num(normalized_matrix)

        return normalized_matrix
    
    def init_sensitive_upstream(self):

        sensitive_edges = set()
        sensitive_system_nodes = set()
        sensitive_user_defined_nodes = set()
        sensitive_nodes = set()

        for i in range(len(self.system_nodes)):
            node = self.system_nodes[i]
            queue = deque()
            queue.append(node)
            sensitive_system_nodes.add(node)
            sensitive_nodes.add(node)

            while queue:
                current_node = queue.popleft()
                for neighbor in self.current_call_graph.predecessors(current_node):
                    if neighbor not in sensitive_nodes:
                        sensitive_nodes.add(neighbor)
                        queue.append(neighbor)
                        if neighbor in self.system_nodes:
                            sensitive_system_nodes.add(neighbor)
                        else:
                            sensitive_user_defined_nodes.add(neighbor)

                    sensitive_edges.add((neighbor, current_node))

        self.sensitive_edges = sensitive_edges
        self.sensitive_user_defined_nodes = sensitive_user_defined_nodes
        self.sensitive_system_nodes = sensitive_system_nodes

    def smail_to_package(self, func_sin, package_list):
        """Map a function signature to its package."""
        class_name = func_sin.split(";")[0][1:].replace("/", ".")
        packages = [class_name.split('.')[0]]
        for part in class_name.split('.')[1:]:
            packages.append(f"{packages[-1]}.{part}")
        packages.reverse()

        for package in packages:
            if package in package_list:
                return package

        # Detect obfuscation if more than half the class components are too short
        if sum(len(part) < 3 for part in class_name.split('.')) > len(class_name.split('.')) / 2:
            return "obfuscated"
        return "self-defined"

    def smail_to_abstract(self, func_sin, abstract_list):
        dim = self.dim
        apigraph_map_file = "entity_class{}.txt".format(dim)
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
        func_package = self.smail_to_package(func_sin, packages[:-2])
        if func_package not in package_class:
            items = func_sin.split(";")[0][1:].replace("/", ".").split('.')
            count_l = 0
            for item in items:
                if len(item) < 3:
                    count_l += 1
            if count_l > (len(items) / 2):
                func_package_class_index = dim + 1
            else:
                func_package_class_index = dim
        else:
            func_package_class_index = package_class[func_package]
        return func_package_class_index



    def process_mutation(self, mutation):
        mutation = mutation.mutation

        type = mutation.get('feature_type', '')
        add_edges = mutation.get('add_edges', [])
        remove_edges = mutation.get('remove_edges', [])
        add_nodes = mutation.get('add_nodes', [])
        remove_nodes = mutation.get('remove_nodes', [])

        state = True
        res = {}

        # Handle the remove node separately
        if type == 'remove_node':
            remove_node = remove_nodes[0]
            if remove_node not in self.nodes and remove_node not in self.user_defined_nodes:
                res["remove_nodes"] = remove_node
                state = False
                return state, res

            remove_edges = mutation.get('remove_edges', [])
            add_edges = mutation.get('add_edges', [])
            for remove_edge in remove_edges:
                remove_edge = tuple(remove_edge)
                if remove_edge in self.edges:
                    self.edges.remove(remove_edge)
                    benign_node = remove_edge[0]
                    benign_node_type = self.node_type(benign_node)
                    end_node = remove_edge[1]
                    end_node_type = self.node_type(end_node)


                    self.apigraph_matrix[benign_node_type][end_node_type] -= 1

            for add_edge in add_edges:
                add_edge = tuple(add_edge)
                if add_edge not in self.edges:
                    self.edges.add(add_edge)
                    benign_node = add_edge[0]
                    benign_node_type = self.node_type(benign_node)
                    end_node = add_edge[1]
                    end_node_type = self.node_type(end_node)


                    self.apigraph_matrix[benign_node_type][end_node_type] += 1
            if remove_node in self.nodes:
                self.nodes.remove(remove_node)
            if remove_node in self.user_defined_nodes:
                self.user_defined_nodes.remove(remove_node)
                type_number = self.node_type(remove_node)

                if type_number == self.dim:
                    if remove_node in self.self_defined_nodes:
                        self.self_defined_nodes.remove(remove_node)

                    if remove_node in self.apigraph_feature_dict[self.dim]:
                        self.apigraph_feature_dict[self.dim].remove(remove_node)


                elif type_number == self.dim+1:
                    if remove_node in self.obfuscated_nodes:
                        self.obfuscated_nodes.remove(remove_node)

                    if remove_node in self.apigraph_feature_dict[self.dim+1]:
                        self.apigraph_feature_dict[self.dim+1].remove(remove_node)
            return state, res

        elif type == 'add_edge':
            add_edge = add_edges[0]
            add_edge = tuple(add_edge)
            if add_edge in self.edges and (add_edge[0] is None or add_edge[1] is None):
                res["add_edges"] = add_edge
                state = False
                return state, res
            else:
                self.edges.add(tuple(add_edge))
                benign_node = add_edge[0]
                benign_node_type = self.node_type(benign_node)
                end_node = add_edge[1]
                end_node_type = self.node_type(end_node)


                self.apigraph_matrix[benign_node_type][end_node_type] += 1

                return state, res

        elif type == 'rewiring':
            remove_edge = remove_edges[0]
            remove_edge = tuple(remove_edge)
            if remove_edge not in self.edges:
                res["remove_edges"] = remove_edge
                state = False
                return state, res
            else:
                if remove_edge in self.edges:
                    self.edges.remove(remove_edge)
                    benign_node = remove_edge[0]
                    benign_node_type = self.node_type(benign_node)
                    end_node = remove_edge[1]
                    end_node_type = self.node_type(end_node)

                    self.apigraph_matrix[benign_node_type][end_node_type] -= 1

                for add_edge in add_edges:
                    add_edge = tuple(add_edge)
                    if add_edge not in self.edges and add_edge[0] is not None and add_edge[1] is not None:
                        self.edges.add(add_edge)
                        benign_node = add_edge[0]
                        benign_node_type = self.node_type(benign_node)
                        end_node = add_edge[1]
                        end_node_type = self.node_type(end_node)


                        self.apigraph_matrix[benign_node_type][end_node_type] += 1

                return state, res

        elif type == 'add_node':
            if len(add_nodes) == 1 or len(add_edges) == 1:
                add_node = add_nodes[0]
                if add_node in self.nodes:
                    res["add_nodes"] = add_node
                    state = False
                    return state, res
                else:
                    self.nodes.add(add_node)
                    self.user_defined_nodes.add(add_node)

                    end_node_type = self.dim
                    if add_node > 30000 and add_node < 50000:
                        self.self_defined_nodes.add(add_node)

                        self.apigraph_feature_dict[self.dim].add(add_node)
                    elif add_node > 50000:
                        self.obfuscated_nodes.add(add_node)

                        self.apigraph_feature_dict[self.dim+1].add(add_node)
                        end_node_type = self.dim + 1
                    self.edges.add(tuple(add_edges[0]))

                    # deal with matrix
                    benign_node = add_edges[0][0]
                    benign_node_type = self.node_type(benign_node)


                    self.apigraph_matrix[benign_node_type][end_node_type] += 1
                    return state, res
            else:  # for dense add node mutation
                for add_node in add_nodes:
                    if add_node in self.nodes:
                        res["add_nodes"] = add_node
                        state = False
                        return state, res

                for add_node in add_nodes:
                    self.nodes.add(add_node)
                    self.user_defined_nodes.add(add_node)
                    self.sensitive_user_defined_nodes.add(add_node)

                for add_edge in add_edges:
                    self.edges.add(tuple(add_edge))
                    self.sensitive_edges.add(add_edge)

                return state, res

        return state, res




    def node_type(self, node):
        if node in self.system_nodes:
            for key, value in self.apigraph_feature_dict.items():
                for v in value:
                    if v == node:

                        return key

            nodes = list(self.original_call_graph.nodes._nodes.keys())
            node_name = nodes[node]
            node_family = self.smail_to_abstract(node_name, self.apigraph_feature_dict.keys())
            self.apigraph_feature_dict[node_family].add(node)

            return node_family
        elif node in self.user_defined_nodes:
            if node in self.self_defined_nodes:
                return self.dim
            elif node in self.obfuscated_nodes:
                return self.dim + 1
            elif node < 50000:
                self.self_defined_nodes.add(node)
                return self.dim
            else:
                self.obfuscated_nodes.add(node)
                return self.dim + 1
        else:
            nodes = list(self.original_call_graph.nodes._nodes.keys())
            if node < len(nodes):
                node_name = nodes[node]
                node_family = self.smail_to_abstract(node_name, self.apigraph_feature_dict.keys())
                self.apigraph_feature_dict[node_family].add(node)
                self.system_nodes.add(node)
                return node_family
            elif node > 30000:
                self.self_defined_nodes.add(node)
                return self.dim
            else:
                self.obfuscated_nodes.add(node)

                return self.dim + 1

    def process_mutation_sequence(self, mutations):
        final_state = True
        for mutation in mutations:
            state, res = self.process_mutation(mutation)
            if not state:
                final_state = False
                break
        return final_state

    def _load(self, file_path, label):
        """Load call graph from a GEXF file."""
        try:
            self.original_call_graph = nx.read_gexf(file_path)
            self.apk_name = file_path.split('/')[-1].split('.gexf')[0]
            self.original_label = label
        except ET.ParseError as e:  # 捕获XML解析错误
            print(f"XML parsing error: {e}")
            return None

    def save(self, file_path, note=None):
        """Save call graph to a GEXF file."""
        if note == None:
            note = ''

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        file_cg = os.path.join(file_path, self.apk_name + '_mutation_' + str(note) + '.gexf')
        if 'ga_failed' in file_path:
            file_cg = os.path.join(file_path,
                                   self.apk_name + '_mutation_' + str(note) + '_failed' + str(time.time()) + '.gexf')

        self.generate_new_call_graph()
        nx.write_gexf(self.current_call_graph, file_cg)


    def cal_apigraph_feature(self):
        """Return the centrality of the current call graph."""
        normalizing_matrix = self.normalizing_matrix()
        return normalizing_matrix

    def generate_new_call_graph(self):
        """Generate a new call graph."""
        self.current_call_graph = nx.DiGraph()
        if self.nodes is None or self.edges is None:
            print("error: nodes or edges is None")
            return False
        self.current_call_graph.add_nodes_from(self.nodes)
        self.current_call_graph.add_edges_from(self.edges)
        return True

    def check_nodes(self):
        if len(self.nodes) != len(self.user_defined_nodes) + len(self.system_nodes):
            print("error: nodes not match")

    def rollback_mutation(self, mutation):
        mutation = mutation.mutation

        add_edges = mutation.get('add_edges', [])
        remove_edges = mutation.get('remove_edges', [])
        add_nodes = mutation.get('add_nodes', [])
        remove_nodes = mutation.get('remove_nodes', [])

        for edge in add_edges:
            edge = tuple(edge)
            if edge in self.edges:
                self.edges.remove(edge)

        for edge in remove_edges:
            edge = tuple(edge)
            if edge not in self.edges:
                self.edges.add(edge)

        for node in add_nodes:
            if node in self.nodes:
                self.nodes.remove(node)
            if node in self.user_defined_nodes:
                self.user_defined_nodes.remove(node)

        for node in remove_nodes:
            if node not in self.nodes:
                self.nodes.add(node)
            if node not in self.user_defined_nodes:
                self.user_defined_nodes.add(node)

    def build_mutation_add_edge(self):
        mutation = Mutation(self.apk_name)
        visited = set()

        if len(self.obfuscated_nodes) == 0:

            user_defined_nodes = list(self.user_defined_nodes)
            keys_list = list(self.apigraph_feature_dict.keys())
            keys_candidate = [key for key in keys_list if len(self.apigraph_feature_dict[key]) != 0]

            end_node_type = random.choice(keys_candidate)
            nodes = list(self.apigraph_feature_dict[end_node_type])
            node1_len = len(user_defined_nodes)
            node2_len = len(nodes)

            if node1_len == 0 or node2_len == 0:
                return None

            try_time = 100
            while node1_len > 0 and node2_len > 0 and try_time > 0:
                node1_idx = random.randint(0, node1_len - 1)
                node2_idx = random.randint(0, node2_len - 1)
                node1 = user_defined_nodes[node1_idx]
                node2 = nodes[node2_idx]

                if (node1, node2) not in visited and (node1, node2) not in self.edges:
                    mutation.mutation['add_edges'].append((node1, node2))
                    mutation.mutation['feature_type'] = 'add_edge'
                    visited.add((node1, node2))
                    return mutation

                if len(visited) == len(self.user_defined_nodes) * len(self.nodes):
                    return None

                try_time -= 1

            return None
        else:
            benign_node_type = random.randint(0, 1)
            user_defined_nodes = list(self.obfuscated_nodes if benign_node_type == 0 else self.self_defined_nodes)

            keys_list = list(self.apigraph_feature_dict.keys())
            keys_candidate = [key for key in keys_list if len(self.apigraph_feature_dict[key]) != 0]

            end_node_type = random.choice(keys_candidate)
            nodes = list(self.apigraph_feature_dict[end_node_type])
            node1_len = len(user_defined_nodes)
            node2_len = len(nodes)

            if node1_len == 0 or node2_len == 0:
                return None

            try_time = 100
            while node1_len > 0 and node2_len > 0 and try_time > 0:
                node1_idx = random.randint(0, node1_len - 1)
                node2_idx = random.randint(0, node2_len - 1)
                node1 = user_defined_nodes[node1_idx]
                node2 = nodes[node2_idx]

                if (node1, node2) not in visited and (node1, node2) not in self.edges:
                    mutation.mutation['add_edges'].append((node1, node2))
                    mutation.mutation['feature_type'] = 'add_edge'
                    visited.add((node1, node2))
                    return mutation

                if len(visited) == len(self.user_defined_nodes) * len(self.nodes):
                    return None

                try_time -= 1

            return None

    def build_mutation_add_edge_self_defined(self):
        mutation = Mutation(self.apk_name)
        visited = set()

        user_defined_nodes = list(self.user_defined_nodes)
        nodes = list(self.nodes)
        node1_len = len(user_defined_nodes)
        node2_len = len(nodes)

        if node1_len == 0 or node2_len == 0:
            return None

        try_time = 100
        while node1_len > 0 and node2_len > 0 and try_time > 0:
            node1_idx = random.randint(0, node1_len - 1)
            node2_idx = random.randint(0, node2_len - 1)
            node1 = user_defined_nodes[node1_idx]
            node2 = nodes[node2_idx]

            if (node1, node2) not in visited and (node1, node2) not in self.edges:
                mutation.mutation['add_edges'].append((node1, node2))
                mutation.mutation['feature_type'] = 'add_edge'
                visited.add((node1, node2))
                return mutation

            if len(visited) == len(self.user_defined_nodes) * len(self.nodes):
                return None

            try_time -= 1

        return None

    def build_mutation_add_edge_sensitive(self):
        mutation = Mutation(self.apk_name)
        visited = set()

        user_defined_nodes = list(self.sensitive_user_defined_nodes)
        nodes = list(self.sensitive_user_defined_nodes.union(self.sensitive_system_nodes))
        node1_len = len(user_defined_nodes)
        node2_len = len(nodes)

        if node1_len == 0 or node2_len == 0:
            return None

        try_time = 100
        while node1_len > 0 and node2_len > 0 and try_time > 0:
            node1_idx = random.randint(0, node1_len - 1)
            node2_idx = random.randint(0, node2_len - 1)
            node1 = user_defined_nodes[node1_idx]
            node2 = nodes[node2_idx]

            if (node1, node2) not in visited and (node1, node2) not in self.edges:
                mutation.mutation['add_edges'].append((node1, node2))
                mutation.mutation['feature_type'] = 'add_edge'
                visited.add((node1, node2))
                return mutation

            if len(visited) == len(self.user_defined_nodes) * len(self.nodes):
                return None

            try_time -= 1

        return None

    def _find_caller_callee(self, node):
        callers = [m for m, n in self.edges if n == node and m != n]
        callees = [m for n, m in self.edges if n == node]
        return callers, callees


    def build_mutation_rewiring(self):
        """
        Build a mutation by rewiring edges:
        1. Randomly select an edge.
        2. Randomly select a node.
        3. Ensure nodes and edges are not duplicated.
        """
        mutation = Mutation(self.apk_name)
        visited_edge = set()
        visited_node = set()

        edges = list(self.edges)
        edge_len = len(edges)

        if len(self.obfuscated_nodes) != 0:
            mid_node_type = random.randint(0, 1)
            if mid_node_type == 0:
                user_defined_nodes = list(self.obfuscated_nodes)
            else:
                user_defined_nodes = list(self.self_defined_nodes)
        else:
            user_defined_nodes = list(self.user_defined_nodes)

        mid_node_len = len(user_defined_nodes)

        if edge_len == 0 or mid_node_len == 0:
            return None

        try_times = 100
        while edge_len > 0 and mid_node_len > 0 and try_times > 0:
            try_times -= 1
            edge_idx = random.randint(0, edge_len - 1)

            edge = edges[edge_idx]
            if edge in visited_edge:
                continue
            visited_edge.add(edge)
            begin_node = edge[0]
            end_node = edge[1]

            if begin_node in self.system_nodes:
                continue

            mid_node_idx = random.randint(0, mid_node_len - 1)
            mid_node = user_defined_nodes[mid_node_idx]
            if mid_node in visited_node:
                continue
            visited_node.add(mid_node)

            mutation.mutation['remove_edges'].append(edge)
            if (begin_node, mid_node) not in self.edges:
                mutation.mutation['add_edges'].append((begin_node, mid_node))

            if (mid_node, end_node) not in self.edges:
                mutation.mutation['add_edges'].append((mid_node, end_node))

            mutation.mutation['feature_type'] = 'rewiring'
            return mutation

        return None  # If no valid edges remain, return None

    def build_mutation_rewiring_sensitive(self):
        """
        Build a mutation by rewiring sensitive edges:
        1. Randomly select an edge.
        2. Randomly select a node.
        3. Ensure no duplicate nodes or edges.
        """
        mutation = Mutation(self.apk_name)
        visited_edge = set()
        visited_node = set()

        edges = list(self.sensitive_edges)
        edge_len = len(edges)
        user_defined_nodes = list(self.user_defined_nodes)
        mid_node_len = len(user_defined_nodes)

        if edge_len == 0 or mid_node_len == 0:
            return None

        try_times = 100
        while edge_len > 0 and mid_node_len > 0 and try_times > 0:
            try_times -= 1
            edge_idx = random.randint(0, edge_len - 1)

            edge = edges[edge_idx]
            if edge in visited_edge:
                continue
            visited_edge.add(edge)
            begin_node = edge[0]
            end_node = edge[1]

            if begin_node in self.system_nodes:
                continue

            mid_node_idx = random.randint(0, mid_node_len - 1)
            mid_node = user_defined_nodes[mid_node_idx]
            if mid_node in visited_node:
                continue
            visited_node.add(mid_node)

            mutation.mutation['remove_edges'].append(edge)
            if (begin_node, mid_node) not in self.edges:
                mutation.mutation['add_edges'].append((begin_node, mid_node))

            if (mid_node, end_node) not in self.edges:
                mutation.mutation['add_edges'].append((mid_node, end_node))

            mutation.mutation['feature_type'] = 'rewiring'
            return mutation

        return None  # Return None if no valid edge is found

    def build_mutation_add_node(self, node_type=-1):
        """
        Build a mutation by adding a node:
        1. Randomly generate a new node.
        2. Randomly select a user-defined or obfuscated node to connect the new node.
        """
        mutation = Mutation(self.apk_name)

        # Randomly generate a new node if type is not provided
        if node_type == -1:
            node_type = random.randint(0, 1)

        # Generate a new self-defined or obfuscated node based on the type
        if node_type == 0:
            new_node = FCG_apigraph.self_defined_number + 30000
            FCG_apigraph.self_defined_number += 1
        else:
            new_node = FCG_apigraph.obfuscated_number + 50000
            FCG_apigraph.obfuscated_number += 1

        # Randomly select an existing node (user-defined or obfuscated) to connect the new node
        if len(self.obfuscated_nodes) != 0:
            benign_node_type = random.randint(0, 1)
            if benign_node_type == 0:
                user_defined_nodes = list(self.user_defined_nodes)
                node_len = len(user_defined_nodes)
                if node_len == 0:
                    return None
                node_idx = random.randint(0, node_len - 1)
                node = user_defined_nodes[node_idx]
            else:
                obfuscated_nodes = list(self.obfuscated_nodes)
                node_len = len(obfuscated_nodes)
                if node_len == 0:
                    return None
                node_idx = random.randint(0, node_len - 1)
                node = obfuscated_nodes[node_idx]
        else:
            user_defined_nodes = list(self.user_defined_nodes)
            node_len = len(user_defined_nodes)
            if node_len == 0:
                return None
            node_idx = random.randint(0, node_len - 1)
            node = user_defined_nodes[node_idx]

        # Add the new node and connect it to the selected existing node
        mutation.mutation['add_nodes'].append(new_node)
        mutation.mutation['add_edges'].append((node, new_node))
        mutation.mutation['feature_type'] = 'add_node'

        return mutation

    def build_mutation_remove_node(self):
        """
        Build a mutation by removing a node:
        1. Randomly select a user-defined or obfuscated node.
        2. Remove all edges connected to the selected node.
        3. Add new edges to maintain graph connectivity if possible.
        """
        mutation = Mutation(self.apk_name)

        # Select user-defined or obfuscated nodes based on their availability
        if len(self.obfuscated_nodes) != 0:
            benign_node_type = random.randint(0, 1)
            if benign_node_type == 0:
                user_defined_nodes = list(self.obfuscated_nodes)
            else:
                user_defined_nodes = list(self.self_defined_nodes)
        else:
            user_defined_nodes = list(self.user_defined_nodes)

        node_len = len(user_defined_nodes)
        try_times = 100

        # Try to find a node to remove within a set number of attempts
        while node_len > 0 and try_times > 0:
            try_times -= 1

            # Randomly select a node
            node_idx = random.randint(0, node_len - 1)
            node = user_defined_nodes[node_idx]

            # Find all incoming (callers) and outgoing (callees) edges for the node
            callers = [m for m, n in self.edges if n == node and m != n]
            callees = [m for n, m in self.edges if n == node]

            # Remove the edges connected to the selected node
            for caller in callers:
                mutation.mutation['remove_edges'].append((caller, node))
            for callee in callees:
                mutation.mutation['remove_edges'].append((node, callee))

            # Mark the node for removal
            mutation.mutation['remove_nodes'].append(node)
            mutation.mutation['feature_type'] = 'remove_node'

            # Add new edges to maintain connectivity if needed
            for caller in callers:
                for callee in callees:
                    if (caller, callee) not in self.edges:
                        mutation.mutation['add_edges'].append((caller, callee))

            return mutation

        return None  # Return None if no valid node for removal is found

    def build_mutation_remove_node_sensitive(self):
        """
        Build a mutation by removing a sensitive node:
        1. Randomly select a sensitive user-defined node.
        2. Remove all edges connected to the selected node.
        3. Add new edges to maintain graph connectivity if possible.
        """
        mutation = Mutation(self.apk_name)
        sensitive_user_defined_nodes = list(self.sensitive_user_defined_nodes)
        node_len = len(sensitive_user_defined_nodes)
        try_times = 100

        # Attempt to find a valid node for removal within a fixed number of tries
        while node_len > 0 and try_times > 0:
            try_times -= 1

            # Randomly select a sensitive node
            node_idx = random.randint(0, node_len - 1)
            node = sensitive_user_defined_nodes[node_idx]

            # Identify all incoming (callers) and outgoing (callees) edges for the node
            callers = [m for m, n in self.edges if n == node and m != n]
            callees = [m for n, m in self.edges if n == node]

            # Remove the edges connected to the selected node
            for caller in callers:
                mutation.mutation['remove_edges'].append((caller, node))

            for callee in callees:
                mutation.mutation['remove_edges'].append((node, callee))

            # Mark the node for removal
            mutation.mutation['remove_nodes'].append(node)
            mutation.mutation['feature_type'] = 'remove_node'

            # Reconnect callers and callees to maintain graph connectivity
            for caller in callers:
                for callee in callees:
                    if (caller, callee) not in self.edges:
                        mutation.mutation['add_edges'].append((caller, callee))

            return mutation

        return None  # Return None if no valid node for removal is found



