from collections import deque
import networkx as nx

class FCG_malscan_tree:
    _framework_prefix = {"Ljava/", "Lsun/", "Landroid/", "Landroidx/", "Lorg/apache/", "Lorg/eclipse/", "Lsoot/",
                       "Ljavax/",
                       "Lcom/google/", "Lorg/xml/", "Ljunit/", "Lorg/json/", "Lorg/w3c/dom/"}

    def __init__(self, file_path, label):
        """Load call graph from a GEXF file."""
        self.raw_call_graph = nx.read_gexf(file_path)
        self.apk_name = file_path.split('/')[-1].split('.gexf')[0]
        self.label = label
        self._init_call_graph()


        assert self.nodes is not None, "nodes is None"
        assert self.edges is not None, "edges is None"
        assert self.numeric_call_graph is not None, "numeric_call_graph is None"
        assert self.user_nodes is not None, "user_nodes is None"
        assert self.system_nodes is not None, "system_nodes is None"

        # Sensitive APIs, originally over 20,000 in malscan, but only 430 here
        assert self.sensitive_apis is not None, "sensitive_apis is None"
        # Bitmap of sensitive APIs, -1 indicates it did not appear, if it did, it corresponds to the ID
        assert self.sensitive_apis_bitmap is not None, "sensitive_apis_bitmap is None"
        # Stores the IDs of sensitive nodes, based on the number of sensitive nodes in the graph
        assert self.sensitive_nodes is not None, "sensitive_nodes is None"
        # Stores the IDs of user nodes that are upstream of sensitive nodes
        assert self.user_critical_nodes is not None, "user_critical_nodes is None"
        # Stores the IDs of system nodes that are upstream of sensitive nodes
        assert self.sys_critical_nodes is not None, "sys_critical_nodes is None"

    def _init_call_graph(self):
        self.sensitive_apis = self._obtain_sensitive_apis()
        # 1. init nodes
        nx_nodes = list(self.raw_call_graph.nodes._nodes.keys())
        nodes = set()
        user_nodes = set()
        system_nodes = set()
        self.last_node_id = len(nx_nodes) - 1
        for i in range(len(nx_nodes)):
            nodes.add(i)
            api_name = nx_nodes[i]
            if not api_name.startswith(tuple(self._framework_prefix)) and api_name not in self.sensitive_apis:
                user_nodes.add(i)
            else:
                system_nodes.add(i)

        node_to_id = {node: idx for idx, node in enumerate(nx_nodes)}

        sensitive_apis_bitmap = []
        sensitive_nodes = []
        for api in self.sensitive_apis:
            if api in nx_nodes:
                sensitive_apis_bitmap.append(node_to_id[api])
                sensitive_nodes.append(node_to_id[api])
            else:
                sensitive_apis_bitmap.append(-1)

        self.nodes = nodes
        self.system_nodes = system_nodes
        self.user_nodes = user_nodes
        # self.node_to_id = node_to_id
        self.sensitive_apis_bitmap = sensitive_apis_bitmap
        self.sensitive_nodes = sensitive_nodes

        # 2. init edges
        nx_edges = list(self.raw_call_graph.edges)
        edges = set()
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

        user_critical_nodes = set()
        sys_critical_nodes = set()
        sensitive_edges = set()
        visited = set()

        for node in self.sensitive_nodes:
            queue = deque()
            queue.append(node)
            visited.add(node)

            while queue:
                current_node = queue.popleft()
                if current_node in self.system_nodes:
                    sys_critical_nodes.add(current_node)
                else:
                    user_critical_nodes.add(current_node)
                for neighbor in self.numeric_call_graph.predecessors(current_node):
                    sensitive_edges.add((neighbor, current_node))
                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)

        self.user_critical_nodes = user_critical_nodes
        self.sys_critical_nodes = sys_critical_nodes
        self.sensitive_edges = sensitive_edges


    def _obtain_sensitive_apis(self):
        file = "sensitive_apis.txt"
        sensitive_apis = []
        with open(file, 'r') as f:
            for line in f.readlines():
                if line.strip() == '':
                    continue
                else:
                    sensitive_apis.append(line.strip())
        return sensitive_apis

    def _cal_degree_feature(self):
        callgraph = self.numeric_call_graph
        N = len(self.nodes)
        vector = []
        unnorm_degree = []

        for node in self.sensitive_apis_bitmap:
            if node == -1:
                vector.append(0)
                unnorm_degree.append(0)
            else:
                d = callgraph.degree(node)
                centrality = d / (N - 1)
                vector.append(centrality)
                unnorm_degree.append(d)
        self.degree_feature = vector
        self.unnorm_degree = unnorm_degree

    def _cal_katz_feature(self):
        alpha = 0.005
        callgraph = self.numeric_call_graph
        degrees = callgraph.degree()
        katz_square_sum = 0
        for _,degree in degrees:
            katz_square_sum += (1+degree*alpha) ** 2
        self.katz_square_sum = katz_square_sum
        katz_centrality = nx.katz_centrality(callgraph, max_iter=5000, alpha=alpha)
        vector = []
        for node in self.sensitive_apis_bitmap:
            if node == -1:
                vector.append(0)
            else:
                vector.append(katz_centrality[node])
        self.katz_feature = vector

    def _cal_closeness_feature(self):
        callgraph = self.numeric_call_graph
        reversed_callgraph = callgraph.reverse(copy=True)
        vector = []

        clossness_extra_details = []
        for node in self.sensitive_apis_bitmap:
            if node == -1:
                vector.append(0)
                clossness_extra_details.append((0,0))
            else:
                sp = nx.single_source_shortest_path(reversed_callgraph, node)
                totsp = sum([len(path) for path in sp.values()])
                len_G = len(reversed_callgraph)
                _closeness_centrality = 0.0
                clossness_extra_details.append((len(sp), totsp))

                if totsp > 0.0 and len_G > 1:
                    _closeness_centrality = (len(sp) - 1.0) / totsp

                    s = (len(sp) - 1.0) / (len_G - 1)
                    _closeness_centrality *= s

                vector.append(_closeness_centrality)
        self.closeness_feature = vector
        self.clossness_extra_details = clossness_extra_details

    def _cal_harmonic_feature(self):
        callgraph = self.numeric_call_graph
        node_centrality = nx.harmonic_centrality(callgraph, nbunch=self.sensitive_nodes)
        vector = []
        for node in self.sensitive_apis_bitmap:
            if node == -1:
                vector.append(0)
            else:
                vector.append(node_centrality[node])
        self.harmonic_feature = vector

    def extract_feature(self, type='degree'):
        self._cal_degree_feature()
        self._cal_katz_feature()
        self._cal_closeness_feature()
        self._cal_harmonic_feature()
        if type == 'degree':
            return self.degree_feature
        elif type == 'katz':
            return self.katz_feature
        elif type == 'closeness':
            return self.closeness_feature
        elif type == 'harmonic':
            return self.harmonic_feature
        elif type == "average":
            average_feature = []
            for i in range(len(self.degree_feature)):
                average_feature.append((self.degree_feature[i] + self.katz_feature[i] + self.closeness_feature[i] + self.harmonic_feature[i]) / 4)
            self.average_feature = average_feature
            return average_feature
        elif type == 'concentrate':
            return self.degree_feature + self.katz_feature + self.closeness_feature + self.harmonic_feature
        else:
            raise ValueError("Invalid feature type: {}".format(type))

    def reduce(self):
        # Merge all user_critical_nodes into a single node
        user_critical_nodes_list = list(self.user_critical_nodes)
        if len(user_critical_nodes_list) == 0:
            self.reserve_user_node = list(self.user_nodes)[0]
            return
        self.reserve_user_node = user_critical_nodes_list[0]
        for node in user_critical_nodes_list[1:]:
            self.nodes.remove(node)
            self.user_critical_nodes.remove(node)

        # Remove outgoing edges from user_critical_nodes
        # If there is an edge from user_critical_nodes to sys_critical_nodes
        # Add an edge from reserve_user_node to sys_critical_nodes
        add_edges = set()
        for edge in self.sensitive_edges:
            if edge[0] in user_critical_nodes_list:
                self.edges.remove(edge)
                if edge[1] in self.sys_critical_nodes:
                    add_edges.add((self.reserve_user_node, edge[1]))
        for edge in add_edges:
            self.edges.add(edge)

        # Update numeric_call_graph
        numeric_call_graph = nx.DiGraph()
        numeric_call_graph.add_nodes_from(self.nodes)
        numeric_call_graph.add_edges_from(self.edges)
        self.numeric_call_graph = numeric_call_graph

    def process_mutations(self, ms):
        root_node = self.reserve_user_node
        # Track the nodes and edges to be added
        add_nodes = []
        add_edges = []

        # Add sparse nodes
        for i in range(ms.add_sparse_nodes_gene):
            node_id = self.last_node_id + 1
            self.last_node_id += 1
            add_nodes.append(node_id)
            # Edges are not added at this stage, as they have minimal impact on feature results but slow down graph computation
            # add_edges.append((root_node, node_id))

        # Add dense nodes
        if ms.add_density_nodes_gene > 0:
            new_nodes = []
            for i in range(ms.add_density_nodes_gene):
                node_id = self.last_node_id + 1
                self.last_node_id += 1
                new_nodes.append(node_id)
                add_nodes.append(node_id)
                add_edges.append((root_node, node_id))
            for i in range(len(new_nodes)):
                for j in range(i, len(new_nodes)):
                    add_edges.append((new_nodes[i], new_nodes[j]))

        # Add long links
        for gene_i in range(len(ms.longlink_gene_detail)):
            # 0 indicates no operation, 1 indicates one operation, n indicates n operations
            if ms.longlink_times_gene[gene_i] == 0:
                continue

            sen_node = ms.longlink_gene_detail[gene_i]
            mid_node_number = ms.longlink_len_gene[gene_i]
            times = ms.longlink_times_gene[gene_i]
            for t in range(int(times)):
                start_node = root_node
                for i in range(mid_node_number):
                    node_id = self.last_node_id + 1
                    self.last_node_id += 1
                    add_nodes.append(node_id)
                    add_edges.append((start_node, node_id))
                    start_node = node_id
                add_edges.append((start_node, sen_node))

        # Update nodes and edges
        self.nodes.update(add_nodes)
        self.edges.update(add_edges)

        # Update numeric_call_graph
        numeric_call_graph = nx.DiGraph()
        numeric_call_graph.add_nodes_from(self.nodes)
        numeric_call_graph.add_edges_from(self.edges)
        self.numeric_call_graph = numeric_call_graph

    def save_gexf(self, path):
        nx.write_gexf(self.numeric_call_graph, path)

