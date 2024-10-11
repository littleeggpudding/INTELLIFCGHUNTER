from collections import deque

import networkx as nx
# 针对malscan的FCG，便于计算4种特征，执行3种mutation
class FCG_malscan_tree:
    _framework_prefix = {"Ljava/", "Lsun/", "Landroid/", "Landroidx/", "Lorg/apache/", "Lorg/eclipse/", "Lsoot/",
                       "Ljavax/",
                       "Lcom/google/", "Lorg/xml/", "Ljunit/", "Lorg/json/", "Lorg/w3c/dom/"}

    def __init__(self, file_path, label):
        """Load call graph from a GEXF file."""
        self.raw_call_graph = nx.read_gexf(file_path)
        # 从file_path里面解析apk_name
        self.apk_name = file_path.split('/')[-1].split('.gexf')[0]  # 最后一个/后面的字符串，去掉.gexf
        self.label = label
        self._init_call_graph()
        assert self.nodes is not None, "nodes is None"
        assert self.edges is not None, "edges is None"
        assert self.numeric_call_graph is not None, "numeric_call_graph is None"
        assert self.user_nodes is not None, "user_nodes is None"
        assert self.system_nodes is not None, "system_nodes is None"

        # 敏感API，原本的malscan是2万多个，这里只有430个
        assert self.sensitive_apis is not None, "sensitive_apis is None"
        # 敏感API的bitmap，-1代表没出现，如果出现了，就是对应的ID
        assert self.sensitive_apis_bitmap is not None, "sensitive_apis_bitmap is None"
        # 哪些node是敏感的，存储的是ID,根据图中有几个敏感节点长度就是几
        assert self.sensitive_nodes is not None, "sensitive_nodes is None"
        # 哪些usernode是敏感节点的上游，存储的是ID
        assert self.user_critical_nodes is not None, "user_critical_nodes is None"
        # 哪些systemnode是敏感节点的上游，存储的是ID
        assert self.sys_critical_nodes is not None, "sys_critical_nodes is None"

    def _init_call_graph(self):
        self.sensitive_apis = self._obtain_sensitive_apis()
        # self.original_call_graph.copy()
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
        # 遍历每条边，将边的节点转换为节点的 ID
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
            # 找到当前敏感节点的所有上游
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
        file = "/data/b/guoqi/malware/resource/important_sensitive_apis.txt"
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
        N = len(self.nodes)  # 获取图中的节点总数
        vector = []
        unnorm_degree = []

        for node in self.sensitive_apis_bitmap:
            if node == -1:
                vector.append(0)
                unnorm_degree.append(0)
            else:
                # 计算节点的度中心性，即节点的度数除以 (N-1)
                d = callgraph.degree(node)
                centrality = d / (N - 1)
                vector.append(centrality)
                unnorm_degree.append(d)
        self.degree_feature = vector
        self.unnorm_degree = unnorm_degree

    def _cal_katz_feature(self):
        alpha = 0.005
        callgraph = self.numeric_call_graph
        degrees = callgraph.in_degree()
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
        reversed_callgraph = callgraph.reverse(copy=True)  # 创建一个反向图视图，以考虑有向图的入度和出度
        vector = []
        # clossness_extra_detail: (len(sp), totsp)
        # len(sp): 从源节点到其他节点的数量
        # totsp: 从源节点到其他节点的最短路径的总长度
        clossness_extra_details = []
        for node in self.sensitive_apis_bitmap:
            if node == -1:
                vector.append(0)
                clossness_extra_details.append((0,0))
            else:
                sp = nx.single_source_shortest_path(reversed_callgraph, node)
                totsp = sum([len(path) for path in sp.values()])  # 计算总距离
                len_G = len(reversed_callgraph)  # 获取图中节点的总数
                _closeness_centrality = 0.0  # 初始化紧密度中心性值为0.0
                clossness_extra_details.append((len(sp), totsp))

                if totsp > 0.0 and len_G > 1:
                    _closeness_centrality = (len(sp) - 1.0) / totsp

                    # 根据 wf_improved 选项进一步标准化紧密度中心性
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
        # 把所有的user_critical_nodes合并成一个点
        user_critical_nodes_list = list(self.user_critical_nodes)
        if len(user_critical_nodes_list) == 0:
            self.reserve_user_node = list(self.user_nodes)[0]
            return
        self.reserve_user_node = user_critical_nodes_list[0]
        for node in user_critical_nodes_list[1:]:
            self.nodes.remove(node)
            self.user_critical_nodes.remove(node)
        # 删除user_critical_nodes出发的边
        # 如果存在从user_critical_nodes到sys_critical_nodes的边
        # 添加一条从reserve_user_node到sys_critical_nodes的边
        # sensitive_edges不维护了
        add_edges = set()
        for edge in self.sensitive_edges:
            if edge[0] in user_critical_nodes_list:
                self.edges.remove(edge)
                if edge[1] in self.sys_critical_nodes:
                    add_edges.add((self.reserve_user_node, edge[1]))
        for edge in add_edges:
            self.edges.add(edge)
        # 更新numeric_call_graph
        numeric_call_graph = nx.DiGraph()
        numeric_call_graph.add_nodes_from(self.nodes)
        numeric_call_graph.add_edges_from(self.edges)
        self.numeric_call_graph = numeric_call_graph

    def process_mutations(self, ms):
        root_node = self.reserve_user_node
        # 统计需要加的点和边
        add_nodes = []
        add_edges = []
        # 添加稀疏点
        for i in range(ms.add_sparse_nodes_gene):
            node_id = self.last_node_id + 1
            self.last_node_id += 1
            add_nodes.append(node_id)
            # 暂时不添加边，因为几乎不影响feature结果，但是会使得图计算变慢
            # add_edges.append((root_node, node_id))
        # 添加稠密点
        if ms.add_density_nodes_gene > 0:
            new_nodes = []
            for i in range(ms.add_density_nodes_gene):
                node_id = self.last_node_id + 1
                self.last_node_id += 1
                new_nodes.append(node_id)
                add_nodes.append(node_id)
                add_edges.append((root_node, node_id))
            for i in range(len(new_nodes)):
                for j in range(i,len(new_nodes)):
                    add_edges.append((new_nodes[i],new_nodes[j]))

        # 添加长边
        for gene_i in range(len(ms.longlink_gene_detail)):
            # 0代表不操作，1代表进行1次操作，n代表进行n次操作
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
        # 更新nodes和edges
        self.nodes.update(add_nodes)
        self.edges.update(add_edges)
        # 更新numeric_call_graph
        numeric_call_graph = nx.DiGraph()
        numeric_call_graph.add_nodes_from(self.nodes)
        numeric_call_graph.add_edges_from(self.edges)
        self.numeric_call_graph = numeric_call_graph
    def save_gexf(self, path):
        nx.write_gexf(self.numeric_call_graph, path)
# 继承FCG类，实现消融实验
class FCG_ABLATION(FCG):
    def __init__(self, file_path, label):
        super(FCG_ABLATION, self).__init__(file_path, label)
    def set_ablation(self, ablation_type):
        assert ablation_type in ['without_density', 'without_longlink', 'without_sparse'], "Invalid ablation type"
        self.ablation_type = ablation_type
    def process_mutations(self, ms):
        assert self.ablation_type in ['without_density', 'without_longlink', 'without_sparse'], "Invalid ablation type"
        if self.ablation_type == 'without_density':
            ms.add_density_nodes_gene = 0
        elif self.ablation_type == 'without_longlink':
            ms.longlink_times_gene = [0] * len(ms.longlink_times_gene)
        elif self.ablation_type == 'without_sparse':
            ms.add_sparse_nodes_gene = 0
        root_node = self.reserve_user_node
        # 统计需要加的点和边
        add_nodes = []
        add_edges = []
        # 添加稀疏点
        for i in range(ms.add_sparse_nodes_gene):
            node_id = self.last_node_id + 1
            self.last_node_id += 1
            add_nodes.append(node_id)
            # 暂时不添加边，因为几乎不影响feature结果，但是会使得图计算变慢
            # add_edges.append((root_node, node_id))
        # 添加稠密点
        if ms.add_density_nodes_gene > 0:
            new_nodes = []
            for i in range(ms.add_density_nodes_gene):
                node_id = self.last_node_id + 1
                self.last_node_id += 1
                new_nodes.append(node_id)
                add_nodes.append(node_id)
                add_edges.append((root_node, node_id))
            for i in range(len(new_nodes)):
                for j in range(i,len(new_nodes)):
                    add_edges.append((new_nodes[i],new_nodes[j]))

        # 添加长边
        for gene_i in range(len(ms.longlink_gene_detail)):
            # 0代表不操作，1代表进行1次操作，n代表进行n次操作
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
        # 更新nodes和edges
        self.nodes.update(add_nodes)
        self.edges.update(add_edges)
        # 更新numeric_call_graph
        numeric_call_graph = nx.DiGraph()
        numeric_call_graph.add_nodes_from(self.nodes)
        numeric_call_graph.add_edges_from(self.edges)
        self.numeric_call_graph = numeric_call_graph