import time
import os
import networkx as nx
import random
from Mutation import Mutation
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np

class FCG_mamadroid:
    number = 0  # Identifier for new nodes to avoid conflicts
    obfuscated_number = 0
    self_defined_number = 0

    def __init__(self, file_path, label, shap_value=None, tree_shap_value=None):
        self.original_call_graph = None  # Initialized once
        self.current_call_graph = None
        self.apk_name = None  # Initialized once

        self.nodes = None  # Updated dynamically
        self.edges = None  # Updated dynamically
        self.user_defined_nodes = None  # Updated dynamically
        self.system_nodes = None  # Fixed
        self.self_defined_nodes = None  # Updated dynamically
        self.obfuscated_nodes = None  # Updated dynamically

        self.original_label = None  # Initialized once

        self.shap_value = shap_value
        self.tree_shap_value = tree_shap_value

        self.mamadroid_feature_dict = {
            'android.': set(),
            'com.google.': set(),
            'java.': set(),
            'javax.': set(),
            'org.xml.': set(),
            'org.apache.': set(),
            'org.junit.': set(),
            'org.json.': set(),
            'org.w3c.dom.': set(),
            'self-defined': set(),
            'obfuscated': set()
        }  # 9 categories, excluding 10: self-defined, 11: obfuscated
        self.mamadroid_matrix = np.zeros((11, 11))

        self._load(file_path, label)

        self._init_call_graph()

    def _init_call_graph(self):
        self.original_call_graph.copy()

        # 1. Initialize nodes
        nodes = list(self.original_call_graph.nodes._nodes.keys())
        nodes_ids = set()
        user_defined_nodes = set()
        system_nodes = set()
        self_defined_nodes = set()
        obfuscated_nodes = set()
        families = list(self.mamadroid_feature_dict.keys())

        for i in range(len(nodes)):
            nodes_ids.add(i)
            current_node = nodes[i]
            current_family = self.smail_to_abstract(current_node, families)
            if current_family == "self-defined":
                user_defined_nodes.add(i)
                self_defined_nodes.add(i)
            elif current_family == "obfuscated":
                user_defined_nodes.add(i)
                obfuscated_nodes.add(i)
            else:
                system_nodes.add(i)
            self.mamadroid_feature_dict[current_family].add(i)

        node_to_id = {node: idx for idx, node in enumerate(nodes)}

        self.nodes = nodes_ids
        self.system_nodes = system_nodes
        self.user_defined_nodes = user_defined_nodes
        self.self_defined_nodes = self_defined_nodes
        self.obfuscated_nodes = obfuscated_nodes

        # 2. Initialize edges
        edges = list(self.original_call_graph.edges)
        edges_ids = set()

        # Traverse each edge and convert nodes to their corresponding IDs
        for edge in edges:
            u, v = edge
            u_id = node_to_id[u]
            v_id = node_to_id[v]
            edge_id = (u_id, v_id)
            edges_ids.add(edge_id)

            u_family = self.smail_to_abstract(u, families)
            v_family = self.smail_to_abstract(v, families)

            u_family_id = list(families).index(u_family)
            v_family_id = list(families).index(v_family)

            self.mamadroid_matrix[u_family_id][v_family_id] += 1

        self.edges = edges_ids

        # Generate the current call graph for serialization
        self.init_sensitive_upstream()

        self.generate_new_call_graph()
        
        
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

    def normalizing_matrix(self):
        row_sums = self.mamadroid_matrix.sum(axis=1)
        row_sums = row_sums[:, np.newaxis]
        normalized_matrix = self.mamadroid_matrix / row_sums
        normalized_matrix = np.nan_to_num(normalized_matrix)
        return normalized_matrix

    def smail_to_abstract(self, func_sin, abstract_list):
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

    def mamadroid_family_feature(self, CG):
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
                caller_family = self.smail_to_abstract(caller, families[:-2])
                callee_family = self.smail_to_abstract(callee, families[:-2])
                caller_family_index = families.index(caller_family)
                callee_family_index = families.index(callee_family)
                markov_family_features[caller_family_index][callee_family_index] += 1

            if total_edges != 0:
                markov_family_features = self.normalizing_matrix(markov_family_features)
                markov_family_features = markov_family_features.flatten()
                return markov_family_features
            else:
                return None

    def process_mutation(self, mutation):
        mutation = mutation.mutation

        type = mutation.get('feature_type', '')
        add_edges = mutation.get('add_edges', [])
        remove_edges = mutation.get('remove_edges', [])
        add_nodes = mutation.get('add_nodes', [])
        remove_nodes = mutation.get('remove_nodes', [])

        state = True
        res = {}

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
                    self.mamadroid_matrix[benign_node_type][end_node_type] -= 1

            for add_edge in add_edges:
                add_edge = tuple(add_edge)
                if add_edge not in self.edges:
                    self.edges.add(add_edge)
                    benign_node = add_edge[0]
                    benign_node_type = self.node_type(benign_node)
                    end_node = add_edge[1]
                    end_node_type = self.node_type(end_node)
                    self.mamadroid_matrix[benign_node_type][end_node_type] += 1

            if remove_node in self.nodes:
                self.nodes.remove(remove_node)
            if remove_node in self.user_defined_nodes:
                self.user_defined_nodes.remove(remove_node)
                type_number = self.node_type(remove_node)
                if type_number == 9:
                    if remove_node in self.self_defined_nodes:
                        self.self_defined_nodes.remove(remove_node)
                    if remove_node in self.mamadroid_feature_dict['self-defined']:
                        self.mamadroid_feature_dict['self-defined'].remove(remove_node)
                elif type_number == 10:
                    if remove_node in self.obfuscated_nodes:
                        self.obfuscated_nodes.remove(remove_node)
                    if remove_node in self.mamadroid_feature_dict['obfuscated']:
                        self.mamadroid_feature_dict['obfuscated'].remove(remove_node)
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
                self.mamadroid_matrix[benign_node_type][end_node_type] += 1
                return state, res

        elif type == 'rewiring':
            remove_edge = remove_edges[0]
            remove_edge = tuple(remove_edge)
            if remove_edge not in self.edges:
                res["remove_edges"] = remove_edge
                print("error: edge not in edges")
                state = False
                return state, res
            else:
                if remove_edge in self.edges:
                    self.edges.remove(remove_edge)
                    benign_node = remove_edge[0]
                    benign_node_type = self.node_type(benign_node)
                    end_node = remove_edge[1]
                    end_node_type = self.node_type(end_node)
                    self.mamadroid_matrix[benign_node_type][end_node_type] -= 1

                for add_edge in add_edges:
                    add_edge = tuple(add_edge)
                    if add_edge not in self.edges and add_edge[0] is not None and add_edge[1] is not None:
                        self.edges.add(add_edge)
                        benign_node = add_edge[0]
                        benign_node_type = self.node_type(benign_node)
                        end_node = add_edge[1]
                        end_node_type = self.node_type(end_node)
                        self.mamadroid_matrix[benign_node_type][end_node_type] += 1

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
                    end_node_type = 9
                    if add_node > 30000 and add_node < 50000:
                        self.self_defined_nodes.add(add_node)
                        self.mamadroid_feature_dict['self-defined'].add(add_node)
                    elif add_node > 50000:
                        self.obfuscated_nodes.add(add_node)
                        self.mamadroid_feature_dict['obfuscated'].add(add_node)
                        end_node_type = 10
                    self.edges.add(tuple(add_edges[0]))

                    benign_node = add_edges[0][0]
                    benign_node_type = self.node_type(benign_node)
                    self.mamadroid_matrix[benign_node_type][end_node_type] += 1
                    return state, res
            else:
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
                    self.sensitive_edges.add(tuple(add_edge))



                return state, res

        return state, res

    def node_type(self, node):
        if node in self.system_nodes:
            key_index = list(self.mamadroid_feature_dict.keys())
            for key, value in self.mamadroid_feature_dict.items():
                for v in value:
                    if v == node:
                        return key_index.index(key)

            nodes = list(self.original_call_graph.nodes._nodes.keys())
            node_name = nodes[node]
            node_family = self.smail_to_abstract(node_name, self.mamadroid_feature_dict.keys())
            self.mamadroid_feature_dict[node_family].add(node)
            return list(self.mamadroid_feature_dict.keys()).index(node_family)
        elif node in self.user_defined_nodes:
            if node in self.self_defined_nodes:
                return 9
            elif node in self.obfuscated_nodes:
                return 10
            elif node < 50000:
                self.self_defined_nodes.add(node)
                return 9
            else:
                self.obfuscated_nodes.add(node)
                return 10
        else:
            nodes = list(self.original_call_graph.nodes._nodes.keys())
            if node < len(nodes):
                node_name = nodes[node]
                node_family = self.smail_to_abstract(node_name, self.mamadroid_feature_dict.keys())
                self.mamadroid_feature_dict[node_family].add(node)
                self.system_nodes.add(node)
                return list(self.mamadroid_feature_dict.keys()).index(node_family)
            elif node > 30000:
                self.self_defined_nodes.add(node)
                return 9
            else:
                self.obfuscated_nodes.add(node)
                return 10


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
        except ET.ParseError as e:
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
            file_cg = os.path.join(file_path, self.apk_name + '_mutation_' + str(note) + '_failed' + str(time.time()) + '.gexf')

        self.generate_new_call_graph()
        nx.write_gexf(self.current_call_graph, file_cg)


    def cal_mamadroid_feature(self):
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
            keys_list = list(self.mamadroid_feature_dict.keys())
            keys_candidate = [key for key in keys_list if len(self.mamadroid_feature_dict[key]) != 0]

            # Randomly select the end node
            end_node_type = random.choice(keys_candidate)
            nodes = list(self.mamadroid_feature_dict[end_node_type])
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

            return None  # Return None if no edges were added
        else:
            benign_node_type = random.randint(0, 1)
            if benign_node_type == 0:
                user_defined_nodes = list(self.obfuscated_nodes)
            else:
                user_defined_nodes = list(self.self_defined_nodes)

            keys_list = list(self.mamadroid_feature_dict.keys())
            keys_candidate = [key for key in keys_list if len(self.mamadroid_feature_dict[key]) != 0]

            # Randomly select the end node
            end_node_type = random.choice(keys_candidate)
            nodes = list(self.mamadroid_feature_dict[end_node_type])
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

            node1_idx = random.randint(0, node1_len-1)
            node2_idx = random.randint(0, node2_len-1)
            node1 = user_defined_nodes[node1_idx]
            node2 = nodes[node2_idx]

            if (node1, node2) not in visited and (node1, node2) not in self.edges:

                mutation.mutation['add_edges'].append((node1, node2))
                mutation.mutation['feature_type'] = 'add_edge'
                visited.add((node1, node2))
                return mutation

            if len(visited) == len(self.user_defined_nodes) * len(self.nodes):
                return None

            try_time = try_time - 1

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

    def build_mutation_rewiring(self):
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

        return None

    def build_mutation_rewiring_sensitive(self):
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

        return None

    def build_mutation_add_node(self, type=-1):
        mutation = Mutation(self.apk_name)
        if type == -1:
            type = random.randint(0, 1)

        if type == 0:
            new_node = FCG_mamadroid.self_defined_number + 30000
            FCG_mamadroid.self_defined_number += 1
        else:
            new_node = FCG_mamadroid.obfuscated_number + 50000
            FCG_mamadroid.obfuscated_number += 1

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

        mutation.mutation['add_nodes'].append(new_node)
        mutation.mutation['add_edges'].append((node, new_node))
        mutation.mutation['feature_type'] = 'add_node'
        return mutation

    def build_mutation_remove_node(self):
        mutation = Mutation(self.apk_name)

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
        while node_len > 0 and try_times > 0:
            try_times -= 1

            node_idx = random.randint(0, node_len - 1)
            node = user_defined_nodes[node_idx]

            callers = [m for m, n in self.edges if n == node and m != n]
            callees = [m for n, m in self.edges if n == node]

            for caller in callers:
                mutation.mutation['remove_edges'].append((caller, node))

            for callee in callees:
                mutation.mutation['remove_edges'].append((node, callee))

            mutation.mutation['remove_nodes'].append(node)
            mutation.mutation['feature_type'] = 'remove_node'

            for caller in callers:
                for callee in callees:
                    if (caller, callee) not in self.edges:
                        mutation.mutation['add_edges'].append((caller, callee))

            return mutation

        return None

    def build_mutation_remove_node_sensitive(self):
        mutation = Mutation(self.apk_name)

        sensitive_user_defined_nodes = list(self.sensitive_user_defined_nodes)
        node_len = len(sensitive_user_defined_nodes)

        try_times = 100
        while node_len > 0 and try_times > 0:
            try_times -= 1

            node_idx = random.randint(0, node_len - 1)
            node = sensitive_user_defined_nodes[node_idx]

            callers = [m for m, n in self.edges if n == node and m != n]
            callees = [m for n, m in self.edges if n == node]

            for caller in callers:
                mutation.mutation['remove_edges'].append((caller, node))

            for callee in callees:
                mutation.mutation['remove_edges'].append((node, callee))

            mutation.mutation['remove_nodes'].append(node)
            mutation.mutation['feature_type'] = 'remove_node'

            for caller in callers:
                for callee in callees:
                    if (caller, callee) not in self.edges:
                        mutation.mutation['add_edges'].append((caller, callee))

            return mutation

        return None

