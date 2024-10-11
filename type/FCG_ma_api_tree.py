import networkx as nx
import numpy as np


class FCG_ma_api_tree:
    # A class to handle call graph operations for Mamadroid and APIGraph models.

    def __init__(self, file_path, label):
        """Load call graph from a GEXF file."""
        self.raw_call_graph = nx.read_gexf(file_path)
        self.apk_name = file_path.split('/')[-1].split('.gexf')[0]
        self.label = label
        self._init_call_graph()
        assert self.nodes is not None, "nodes is None"
        assert self.edges is not None, "edges is None"
        assert self.numeric_call_graph is not None, "numeric_call_graph is None"

    def _init_call_graph(self):
        # 1. init nodes
        nx_nodes = list(self.raw_call_graph.nodes._nodes.keys())
        nodes = set()
        for i in range(len(nx_nodes)):
            nodes.add(i)
        self.last_node_id = len(nx_nodes) - 1
        node_to_id = {node: idx for idx, node in enumerate(nx_nodes)}
        self.id_to_node = {idx: node for idx, node in enumerate(nx_nodes)}
        self.nodes = nodes

        # 2. init edges
        nx_edges = list(self.raw_call_graph.edges)
        edges = set()
        # Iterate over each edge, converting the node of the edge to the node ID
        for nx_edge in nx_edges:
            u, v = nx_edge
            u_id = node_to_id[u]
            v_id = node_to_id[v]
            edge = (u_id, v_id)
            edges.add(edge)
        self.edges = edges

        numeric_call_graph = nx.DiGraph()
        numeric_call_graph.add_nodes_from(nodes)
        numeric_call_graph.add_edges_from(edges)
        self.numeric_call_graph = numeric_call_graph

    def _smail_to_abstract(self, func_sin, abstract_list):
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

    def _smail_to_package(self, func_sin, package_list):
        class_name = func_sin.split(";")[0][1:].replace("/", ".")

        items = class_name.split('.')
        packages = []
        packages.append(items[0])
        for i in range(1, len(items)):
            cur_package = packages[i - 1] + '.' + items[i]
            packages.append(cur_package)

        packages.reverse()
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

    def _normalizing_matrix(self, call_matrix):
        row_sums = call_matrix.sum(axis=1)
        row_sums = row_sums[:, np.newaxis]
        epsilon = 1e-10
        row_sums += epsilon
        normalized_matrix = call_matrix / row_sums
        normalized_matrix = np.nan_to_num(normalized_matrix)

        return normalized_matrix

    def _extract_mamadroid_feature(self):
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


        node_to_family = {}
        for node in self.nodes:
            node_sig = self.id_to_node[node]
            node_family = self._smail_to_abstract(node_sig, families[:-2])
            family_index = families.index(node_family)
            family_dict[family_index].add(node)
            node_to_family[node] = family_index

        family_count = {}
        for edge in self.edges:
            caller = edge[0]
            callee = edge[1]
            caller_family_index = node_to_family[caller]
            callee_family_index = node_to_family[callee]
            detail_edges[caller_family_index][callee_family_index] += 1

        for i in range(11):
            family_count[i] = len(family_dict[i])

        markov_family_features = self._normalizing_matrix(detail_edges)
        markov_family_features = markov_family_features.flatten()
        self.feature = markov_family_features
        self.detail_edges = detail_edges
        self.type_count = family_count
        self.type_dict = family_dict

    def _extract_apigraph_feature(self):
        dim = 50
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
        detail_edges = np.zeros((dim + 2, dim + 2))

        apigraph_dict = {}
        for i in range(dim + 2):
            apigraph_dict[i] = set()
        node_to_type = {}
        for node in self.nodes:
            node_sig = self.id_to_node[node]
            node_package = self._smail_to_package(node_sig, packages[:-2])
            if node_package in package_class:
                node_package_class_index = package_class[node_package]
                apigraph_dict[node_package_class_index].add(node)
                node_to_type[node] = node_package_class_index
        for edge in self.edges:
            caller = edge[0]
            callee = edge[1]
            if caller not in node_to_type or callee not in node_to_type:
                continue
            caller_package_class_index = node_to_type[caller]
            callee_package_class_index = node_to_type[callee]
            detail_edges[caller_package_class_index][callee_package_class_index] += 1
        apigraph_count = {}
        for i in range(dim + 2):
            apigraph_count[i] = len(apigraph_dict[i])
        markov_apigraph_features = self._normalizing_matrix(detail_edges)
        markov_apigraph_features = markov_apigraph_features.flatten()

        self.feature = markov_apigraph_features
        self.detail_edges = detail_edges
        self.type_count = apigraph_count
        self.type_dict = apigraph_dict

    def extract_features(self,feature_type):
        assert feature_type in ['apigraph', 'mamadroid']
        if feature_type == 'apigraph':
            self._extract_apigraph_feature()
        else:
            self._extract_mamadroid_feature()
        return self.feature
