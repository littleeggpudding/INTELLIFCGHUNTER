import sys
import os
import time

sys.path.append(os.path.abspath('../task'))

from ExtractFeature import obtain_sensitive_apis, extract_degree_centrality_from_CG, \
    extract_katz_centrality_from_CG, extract_katz_centrality_from_CG_new_pytorch, extract_harmonic_centrality_from_CG, extract_closeness_centrality_from_CG, extract_closeness_harmonic_centrality_from_CG, extract_harmonic_centrality_from_CG_new, extract_closeness_centrality_from_CG_new
import os
import networkx as nx
import random
from Mutation import Mutation
from collections import deque
import xml.etree.ElementTree as ET
import torch
import numpy as np



# call_graph	APK的图的信息,是networkx.DiGraph()
# process_mutation	根据Mutation更新call_graph
# process_mutation_sequence	根据Mutation sequence更新call_graph
# load	从gexf文件读取
# save	写到gexf文件
# get_degree_centrality	提取degree特征
# get_katz_centrality

# 1. load
# 2. init_call_graph
# 3. init_centralities
# 4. process_mutation
# 5. process_mutation_sequence
# 6. save

class FCG:
    number = 0  # 新加节点的编号, 防止冲突

    def __init__(self, file_path, label, shap_value=None, tree_shap_value=None):
        self.original_call_graph = None # 初始化一次
        self.current_call_graph = None
        self.apk_name = None # 初始化一次

        self.nodes = None  # 实时更新
        self.edges = None  # 实时更新
        self.user_defined_nodes = None  # 实时更新
        self.system_nodes = None  # 实时更新
        self.boundary_edges = None  # 实时更新， user-->system,jan21已经不用了
        self.sensitive_nodes = None  # 实时更新 一个2万多的list

        #sensitive可达的区域
        self.sensitive_system_nodes = None  # 实时更新
        self.sensitive_user_defined_nodes = None  # 实时更新
        self.sensitive_edges = None  # 实时更新

        self.used_sensitive_nodes = None  # 初始化一次，sensitive_nodes里面不为-1的节点
        self.original_label = None # 初始化一次

        self.degree_feature = []
        self.katz_feature = []
        self.closeness_feature = []
        self.harmonic_feature = []

        self.shap_value = shap_value
        self.tree_shap_value = tree_shap_value

        self.adj_matrix = None



        self._framework = {"Ljava/", "Lsun/", "Landroid/", "Landroidx/", "Lorg/apache/", "Lorg/eclipse/", "Lsoot/",
                           "Ljavax/",
                           "Lcom/google/", "Lorg/xml/", "Ljunit/", "Lorg/json/", "Lorg/w3c/dom/"}

        self._framework_java = {"java.","sun.", "android.", "androidx.", "org.apache.", "org.eclipse.", "soot.", "javax.",
                                "com.google.", "org.xml.", "junit.", "org.json.", "org.w3c.dom."}


        self._apis = obtain_sensitive_apis()  # 将其转化为集合以提高查找效率

        self._load(file_path, label)

        self._init_call_graph()

    def _init_call_graph(self):
        self.original_call_graph.copy()
        # 首先获取所有节点，如果还没有获取过的话
        # self.nodes = list(self.original_call_graph.nodes._nodes.keys())
        # self.nodes = list(self.original_call_graph.nodes)
        #1. init nodes
        nodes = list(self.original_call_graph.nodes._nodes.keys())
        nodes_ids = set()
        user_defined_nodes = set()
        system_nodes = set()
        sensitive_nodes = []
        for i in range(len(nodes)):
            nodes_ids.add(i)
            # print(nodes[i])
            current_node = nodes[i]
            if not current_node.startswith(tuple(self._framework)) and current_node not in self._apis:
                user_defined_nodes.add(i)
            else:
                system_nodes.add(i)

        node_to_id = {node: idx for idx, node in enumerate(nodes)}

        used_sensitive_nodes = []
        for api in self._apis:
            if api in nodes:
                sensitive_nodes.append(node_to_id[api])
                used_sensitive_nodes.append(node_to_id[api])
            else:
                sensitive_nodes.append(-1)

        self.nodes = nodes_ids
        self.sensitive_nodes = sensitive_nodes
        self.system_nodes = system_nodes
        self.user_defined_nodes = user_defined_nodes
        self.used_sensitive_nodes = used_sensitive_nodes



        #2. init edges
        edges = list(self.original_call_graph.edges)
        edges_ids = set()
        boundary_edges = set()
        # 遍历每条边，将边的节点转换为节点的 ID
        for edge in edges:
            u, v = edge
            u_id = node_to_id[u]
            v_id = node_to_id[v]
            edge_id = (u_id, v_id)
            edges_ids.add(edge_id)
            if u_id in self.user_defined_nodes and v_id in self.system_nodes:
                boundary_edges.add(edge_id)
        self.edges = edges_ids

        # 如果用户定义的节点和系统节点还没有被定义，那么一次遍历就可以分别赋值
        # if self.user_defined_nodes is None or self.system_nodes is None:
        #     self.user_defined_nodes = []
        #     self.system_nodes = []
        #
        #     for node in self.nodes:
        #         if not node.startswith(tuple(self._framework)) and node not in self._apis:
        #             self.user_defined_nodes.append(node)
        #         else:
        #             self.system_nodes.append(node)
        #
        # # # 获取边缘和边界边缘，如果还没有获取过的话
        # if self.edges is None or self.boundary_edges is None:
        #     self.edges = []
        #     self.boundary_edges = []
        #
        #     for node1, node2 in self.original_call_graph.edges:
        #         self.edges.append((node1, node2))
        #         if node1 in self.user_defined_nodes and node2 in self.system_nodes:
        #             self.boundary_edges.append((node1, node2))
        #
        #
        #
        # self._init_centralities()

        # current_call_graph, 序列化的时候用
        self.generate_new_call_graph()

        self.init_sensitive_upstream()
        # print("Init call graph finished!!!!")
        
        
    def _init_call_graph_for_gcn(self):
        self.original_call_graph.copy()
        # 首先获取所有节点，如果还没有获取过的话
        # self.nodes = list(self.original_call_graph.nodes._nodes.keys())
        # self.nodes = list(self.original_call_graph.nodes)
        #1. init nodes
        nodes = list(self.original_call_graph.nodes._nodes.keys())
        nodes_ids = set()
        user_defined_nodes = set()
        system_nodes = set()
        sensitive_nodes = []
        for i in range(len(nodes)):
            nodes_ids.add(i)
            # print(nodes[i])
            # current_node = nodes[i]
            # if not current_node.startswith(tuple(self._framework)) and current_node not in self._apis:
            #     user_defined_nodes.add(i)
            # else:
            #     system_nodes.add(i)

        node_to_id = {node: idx for idx, node in enumerate(nodes)}
        #
        # used_sensitive_nodes = []
        # for api in self._apis:
        #     if api in nodes:
        #         sensitive_nodes.append(node_to_id[api])
        #         used_sensitive_nodes.append(node_to_id[api])
        #     else:
        #         sensitive_nodes.append(-1)

        self.nodes = nodes_ids
        # self.sensitive_nodes = sensitive_nodes
        # self.system_nodes = system_nodes
        # self.user_defined_nodes = user_defined_nodes
        # self.used_sensitive_nodes = used_sensitive_nodes



        #2. init edges
        edges = list(self.original_call_graph.edges)
        edges_ids = set()
        # boundary_edges = set()
        # 遍历每条边，将边的节点转换为节点的 ID
        for edge in edges:
            u, v = edge
            u_id = node_to_id[u]
            v_id = node_to_id[v]
            edge_id = (u_id, v_id)
            edges_ids.add(edge_id)
            # if u_id in self.user_defined_nodes and v_id in self.system_nodes:
            #     boundary_edges.add(edge_id)
        self.edges = edges_ids

        # 如果用户定义的节点和系统节点还没有被定义，那么一次遍历就可以分别赋值
        # if self.user_defined_nodes is None or self.system_nodes is None:
        #     self.user_defined_nodes = []
        #     self.system_nodes = []
        #
        #     for node in self.nodes:
        #         if not node.startswith(tuple(self._framework)) and node not in self._apis:
        #             self.user_defined_nodes.append(node)
        #         else:
        #             self.system_nodes.append(node)
        #
        # # # 获取边缘和边界边缘，如果还没有获取过的话
        # if self.edges is None or self.boundary_edges is None:
        #     self.edges = []
        #     self.boundary_edges = []
        #
        #     for node1, node2 in self.original_call_graph.edges:
        #         self.edges.append((node1, node2))
        #         if node1 in self.user_defined_nodes and node2 in self.system_nodes:
        #             self.boundary_edges.append((node1, node2))
        #
        #
        #
        # self._init_centralities()

        # current_call_graph, 序列化的时候用
        self.generate_new_call_graph()
        print("Init call graph finished!!!!")

    def count_and_sort_upstream_nodes(self, all_sensitive_upstream):
        #可以统计初始状态下的，每一个上游节点能调用的下游节点的个数
        # 创建一个字典来存储上游节点的频数
        frequency_dict = {}

        all_sensitive_edges = []

        # 遍历字典中的每个敏感节点和它的上游节点列表
        for upstream in all_sensitive_upstream.values():
            upstream_nodes = upstream.get('nodes', [])
            for node in upstream_nodes:
                # 对每个上游节点的出现次数进行计数
                if node in frequency_dict:
                    frequency_dict[node] += 1
                else:
                    frequency_dict[node] = 1

            upstream_edges = upstream.get('edges', [])
            all_sensitive_edges.extend(upstream_edges)

        # print("frequency_dict", frequency_dict)
        # 将节点按照频数进行排序，返回一个排序后的列表
        sorted_nodes = sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True)
        # print("sorted_nodes", sorted_nodes)
        return sorted_nodes, all_sensitive_edges


    def filter_used_sensitive_apis(self, vector):
        non_zero_vector = []
        if vector is None:
            return non_zero_vector
        for i in range(len(vector)):
            if vector[i] != 0:
                non_zero_vector.append(self._apis[i])

        print("non_zero_vector", len(non_zero_vector))
        return non_zero_vector


    def process_mutation(self, mutation):
        # print("process mutation start!!!!")
        mutation = mutation.mutation

        type = mutation.get('feature_type', '')
        add_edges = mutation.get('add_edges', [])
        remove_edges = mutation.get('remove_edges', [])
        add_nodes = mutation.get('add_nodes', [])
        remove_nodes = mutation.get('remove_nodes', [])

        # print("mutation", mutation)

        state = True
        res = {}

        #单独处理remove_node
        if type == 'remove_node':
            remove_node = remove_nodes[0]
            if remove_node not in self.nodes and remove_node not in self.user_defined_nodes:
                res["remove_nodes"] = remove_node
                # print("error: node not in nodes")
                state = False
                return state, res

            flag = False
            if remove_node in self.sensitive_user_defined_nodes:
                self.sensitive_user_defined_nodes.remove(remove_node)
                flag = True

            #renew caller and callee
            # new_callers = [m for m, n in self.edges if n == remove_node and m != n]  # 有的边是自己指向自己的，包括在calleee中，就不用包括在caller中
            # new_callees = [m for n, m in self.edges if n == remove_node]
            # for caller in new_callers:
            #     for callee in new_callees:
            #         if caller != callee and (caller, callee) not in self.edges and caller is not None and callee is not None:
            #             self.edges.add((caller, callee))
            #             mutation['add_edges'].append((caller, callee))
            #             if flag:
            #                 self.sensitive_edges.add((caller, callee))
            #
            # for caller in new_callers:
            #     if (caller, remove_node) in self.edges:
            #         self.edges.remove((caller, remove_node))
            #         mutation['remove_edges'].append((caller, remove_node))
            #         if flag and (caller, remove_node) in self.sensitive_edges:
            #             self.sensitive_edges.remove((caller, remove_node))
            #
            # for callee in new_callees:
            #     if (remove_node, callee) in self.edges:
            #         self.edges.remove((remove_node, callee))
            #         mutation['remove_edges'].append((remove_node, callee))
            #         if flag and (remove_node, callee) in self.sensitive_edges:
            #             self.sensitive_edges.remove((remove_node, callee))

            #不及时更新caller和callee
            remove_edges = mutation.get('remove_edges', [])
            add_edges = mutation.get('add_edges', [])
            for remove_edge in remove_edges:
                remove_edge = tuple(remove_edge)
                if remove_edge in self.edges:
                    self.edges.remove(remove_edge)
                    if flag and remove_edge in self.sensitive_edges:
                        self.sensitive_edges.remove(remove_edge)

            for add_edge in add_edges:
                add_edge = tuple(add_edge)
                if add_edge not in self.edges:
                    self.edges.add(add_edge)
                    if flag:
                        self.sensitive_edges.add(add_edge)

            self.nodes.remove(remove_node)
            if remove_node in self.user_defined_nodes:
                self.user_defined_nodes.remove(remove_node)
            return state, res

        elif type == 'add_edge':
            add_edge = add_edges[0]
            add_edge = tuple(add_edge)
            if add_edge in self.edges and (add_edge[0] is None or add_edge[1] is None):
                res["add_edges"] = add_edge
                # print("error: edge already in edges")
                state = False
                return state, res
            else:
                self.edges.add(tuple(add_edge))
                # update sensitive area
                if add_edge[1] in self.sensitive_user_defined_nodes or add_edge[1] in self.sensitive_system_nodes:
                    #把begin node 的 upstream加入
                    begin_node = add_edge[0]
                    upstream, upstream_edges = self.check_node_upstream(begin_node)
                    self.sensitive_user_defined_nodes.update(upstream)
                    self.sensitive_edges.update(upstream_edges)

                return state, res

        elif type == 'rewiring':
            remove_edge = remove_edges[0]
            remove_edge = tuple(remove_edge)
            if remove_edge not in self.edges and remove_edge not in self.sensitive_edges:
            # if remove_edge not in self.edges:
                res["remove_edges"] = remove_edge
                print("error: edge not in edges")
                state = False
                return state, res
            else:
                if remove_edge in self.edges:
                    self.edges.remove(remove_edge)
                if remove_edge in self.sensitive_edges:
                    self.sensitive_edges.remove(remove_edge)

                flag = False
                if remove_edge[1] in self.sensitive_user_defined_nodes or remove_edge[1] in self.sensitive_system_nodes:
                    flag = True

                for add_edge in add_edges:
                    add_edge = tuple(add_edge)
                    if add_edge not in self.edges and add_edge[0] is not None and add_edge[1] is not None:
                        self.edges.add(add_edge)
                        if flag:
                            self.sensitive_edges.add(add_edge)

                if flag:
                    mid_node = ''
                    if len(add_edges) == 1:
                        if add_edges[0][0] == remove_edge[0]:
                            mid_node = add_edges[0][1]
                        elif add_edges[0][1] == remove_edge[1]:
                            mid_node = add_edges[0][0]
                    elif len(add_edges) == 2:
                        mid_node = add_edges[0][1]

                    if mid_node != '':
                        upstream, upstream_edges = self.check_node_upstream(mid_node)
                        self.sensitive_user_defined_nodes.update(upstream)
                        self.sensitive_edges.update(upstream_edges)

                return state, res

        elif type == 'add_node':
            if len(add_nodes) == 1 or len(add_edges) == 1:
                add_node = add_nodes[0]
                if add_node in self.nodes:
                    res["add_nodes"] = add_node
                    # print("error: node already in nodes")
                    state = False
                    return state, res
                else:
                    self.nodes.add(add_node)
                    self.user_defined_nodes.add(add_node)
                    self.edges.add(tuple(add_edges[0]))
                    return state, res
            else:
                for add_node in add_nodes:
                    if add_node in self.nodes:
                        res["add_nodes"] = add_node
                        # print("error: node already in nodes")
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



    def process_mutation_sequence(self, mutations):  # mutations是一个list，里面存储的是Mutation
        # print("process mutation sequence start!!!!")
        # print("mutation sequence len:", len(mutations))
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
            # 从file_path里面解析apk_name
            self.apk_name = file_path.split('/')[-1].split('.gexf')[0]  # 最后一个/后面的字符串，去掉.gexf
            self.original_label = label
            print("load the original FCG finished!!!!")
        except ET.ParseError as e:  # 捕获XML解析错误
            print(f"XML parsing error: {e}")
            return None


    def save(self, file_path, note=None):
        """Save call graph to a GEXF file."""
        if note == None:
            note = ''

        # 如果没有这个文件夹，就创建一个
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        file_cg = os.path.join(file_path, self.apk_name + '_mutation_' + str(note) + '.gexf')
        if 'ga_failed' in file_path:
            file_cg = os.path.join(file_path, self.apk_name + '_mutation_' + str(note) + '_failed' + str(time.time()) + '.gexf')

        self.generate_new_call_graph()  # 保证存储的是最新的call graph
        nx.write_gexf(self.current_call_graph, file_cg)
        print("save the latest FCG finished!!!!")

    
    # 每次mutation之后都要重新计算
    def cal_centralities(self, type='all'):
        """Return the centrality of the current call graph."""
        # 用networkx生成
        state = self.generate_new_call_graph()
        # print("generate new call graph ", state)
        if not state:
            return False

        try:

            if type == 'all':
                self.degree_feature = extract_degree_centrality_from_CG(self.current_call_graph, self.sensitive_nodes)
                self.katz_feature = extract_katz_centrality_from_CG(self.current_call_graph,  self.sensitive_nodes)
                self.closeness_feature = extract_closeness_centrality_from_CG(self.current_call_graph,  self.sensitive_nodes)
                self.harmonic_feature = extract_harmonic_centrality_from_CG_new(self.current_call_graph,  self.sensitive_nodes, self.used_sensitive_nodes)
                self.closeness_feature = extract_closeness_centrality_from_CG_new(self.current_call_graph,  self.sensitive_nodes)
                # self.closeness_feature, self.harmonic_feature = extract_closeness_harmonic_centrality_from_CG(self.current_call_graph,  self.sensitive_nodes)
                # print(self.closeness_feature)
                return True

            elif type == 'degree':
                self.degree_feature = extract_degree_centrality_from_CG(self.current_call_graph,  self.sensitive_nodes)
                return True

            elif type == 'katz':
                start = time.time()
                self.katz_feature = extract_katz_centrality_from_CG(self.current_call_graph,  self.sensitive_nodes)
                # res = extract_katz_centrality_from_CG_new_pytorch(self.current_call_graph,  self.sensitive_nodes)
                # print("time:", time.time() - start)
                # #check
                # cnt = 0
                # cnt_non_zero = 0
                # for i in range(len(self.katz_feature)):
                #     if self.katz_feature[i] != res[i]:
                #         print(f"error: {i} {self.katz_feature[i]} {res[i]}")
                #         cnt += 1
                #     if self.katz_feature[i] != 0:
                #         cnt_non_zero += 1
                #
                # print(f"error cnt: {cnt}")
                # print(f"non_zero cnt: {cnt_non_zero}")
                # print("new katz!!!")
                return True

            elif type == 'closeness':
                self.closeness_feature = extract_closeness_centrality_from_CG_new(self.current_call_graph,  self.sensitive_nodes)
                return True

            elif type == 'harmonic':
                self.harmonic_feature = extract_harmonic_centrality_from_CG_new(self.current_call_graph,  self.sensitive_nodes, self.used_sensitive_nodes)
                return True

        except Exception as e:
            print(f"An error occurred: {e}")
            return False



    def generate_new_call_graph(self):
        """Generate a new call graph."""
        # print(len(self.nodes), len(self.edges))
        self.current_call_graph = nx.DiGraph()
        if self.nodes is None or self.edges is None:
            print("error: nodes or edges is None")
            return False
        self.current_call_graph.add_nodes_from(self.nodes)
        self.current_call_graph.add_edges_from(self.edges)
        # edge_index_for_gcn 是符合PyTorch Geometric的格式，shape为[2, num_edges]
        # self.edge_index_for_gcn = torch.tensor([2, len(self.edges)], dtype=torch.long)
        # for edge in self.edges:
        #     self.edge_index_for_gcn.append([edge[0], edge[1]])
        return True


    def check_nodes(self):
        if len(self.nodes) != len(self.user_defined_nodes) + len(self.system_nodes):
            print("error: nodes not match")

    def rollback_mutation(self, mutation):
        # print("rollback mutation start!!!!")
        # print(mutation)
        # #mutation是一个字典，add_edges, remove_edges, add_nodes, remove_nodes
        # print("FCG information before mutation:")
        # print("nodes:", len(self.nodes))
        # print("edges:", len(self.edges))

        mutation = mutation.mutation

        add_edges = mutation.get('add_edges', [])
        remove_edges = mutation.get('remove_edges', [])
        add_nodes = mutation.get('add_nodes', [])
        remove_nodes = mutation.get('remove_nodes', [])

        for edge in add_edges:
            edge = tuple(edge)
            if edge in self.edges:
                self.edges.remove(edge)
            # if edge in self.boundary_edges:
            #     self.boundary_edges.remove(edge)

        for edge in remove_edges:
            edge = tuple(edge)
            if edge not in self.edges:
                self.edges.add(edge)
            # if edge not in self.boundary_edges:
            #     self.boundary_edges.append(edge)

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

        # print("rollback mutation finished!!!!")

    def build_mutation_add_edge(self):
        # print("add edge")
        # user -> system
        mutation = Mutation(self.apk_name)
        visited = set()

        # cannot_user_defined_nodes = set()
        # cannot_system_nodes = set()
        #
        # for edge in self.boundary_edges:
        #     cannot_user_defined_nodes.add(edge[0])
        #     cannot_system_nodes.add(edge[1])
        #
        # can_user_defined_nodes = set(self.user_defined_nodes) - cannot_user_defined_nodes
        # can_system_nodes = set(self.system_nodes) - cannot_system_nodes

        # print("boundary edges", len(self.boundary_edges))

        user_defined_nodes = list(self.user_defined_nodes)
        nodes = list(self.nodes)
        node1_len = len(user_defined_nodes)
        node2_len = len(nodes)

        if node1_len == 0 or node2_len == 0:
            return None

        try_time = 100
        while node1_len > 0 and node2_len > 0 and try_time > 0:
            # node1 = random.choice(list(can_user_defined_nodes))
            # node2 = random.choice(list(can_system_nodes))

            node1_idx = random.randint(0, node1_len-1)
            node2_idx = random.randint(0, node2_len-1)
            node1 = user_defined_nodes[node1_idx]
            node2 = nodes[node2_idx]

            if (node1, node2) not in visited and (node1, node2) not in self.edges:
                # print("add edge:", node1, node2)

                # clear mutation
                # mutation.clear_mutation()
                mutation.mutation['add_edges'].append((node1, node2))  # 存储的时候是list，在fcg也更新了nodes和edges
                mutation.mutation['feature_type'] = 'add_edge'
                # mutation.save_log()
                visited.add((node1, node2))
                # print("add edge finished!!!!")
                return mutation

            if len(visited) == len(self.user_defined_nodes) * len(self.nodes):
                return None

            try_time = try_time - 1

        return None  # 如果没有添加任何边，返回-1

    def build_mutation_add_edge_sensitive(self):
        # print("add edge")
        # user -> system
        mutation = Mutation(self.apk_name)
        visited = set()

        # cannot_user_defined_nodes = set()
        # cannot_system_nodes = set()
        #
        # for edge in self.boundary_edges:
        #     cannot_user_defined_nodes.add(edge[0])
        #     cannot_system_nodes.add(edge[1])
        #
        # can_user_defined_nodes = set(self.user_defined_nodes) - cannot_user_defined_nodes
        # can_system_nodes = set(self.system_nodes) - cannot_system_nodes

        # print("boundary edges", len(self.boundary_edges))

        # user_defined_nodes = list(self.user_defined_nodes)
        user_defined_nodes = list(self.sensitive_user_defined_nodes)
        #末尾在就可以
        nodes = list(self.sensitive_user_defined_nodes.union(self.sensitive_system_nodes))
        node1_len = len(user_defined_nodes)
        node2_len = len(nodes)

        if node1_len == 0 or node2_len == 0:
            return None

        try_time = 100
        while node1_len > 0 and node2_len > 0 and try_time > 0:
            # node1 = random.choice(list(can_user_defined_nodes))
            # node2 = random.choice(list(can_system_nodes))

            node1_idx = random.randint(0, node1_len-1)
            node2_idx = random.randint(0, node2_len-1)
            node1 = user_defined_nodes[node1_idx]
            node2 = nodes[node2_idx]

            if (node1, node2) not in visited and (node1, node2) not in self.edges:
                # print("add edge:", node1, node2)

                # clear mutation
                # mutation.clear_mutation()
                mutation.mutation['add_edges'].append((node1, node2))  # 存储的时候是list，在fcg也更新了nodes和edges
                mutation.mutation['feature_type'] = 'add_edge'
                # mutation.save_log()
                visited.add((node1, node2))
                # print("add edge finished!!!!")
                return mutation

            if len(visited) == len(self.user_defined_nodes) * len(self.nodes):
                return None

            try_time = try_time - 1

        return None  # 如果没有添加任何边，返回-1


    def build_mutation_add_edge_type1(self):
        #找两个节点，都没有非敏感节点下游

        # print("add edge")
        # user -> system
        mutation = Mutation(self.apk_name)
        visited = set(self.boundary_edges)

        #补集
        can_user_defined_nodes = set(self.user_defined_nodes) - set(self.sensitive_user_defined_nodes)
        can_user_defined_nodes = list(can_user_defined_nodes)
        
        can_system_nodes = set(self.system_nodes) - set(self.sensitive_system_nodes)
        can_system_nodes = list(can_system_nodes)


        if len(can_user_defined_nodes) == 0:
            return None
        else:
            target1 = can_user_defined_nodes

        target2 = can_system_nodes
        #target = can_user_defined_nodes

        label = 'add_edge_type1'
        while True:
            node1 = random.choice(target1)

            node2 = random.choice(target2)

            if (node1, node2) not in visited:
                # print("add edge:", node1, node2)

                # clear mutation
                # mutation.clear_mutation()
                mutation.mutation['add_edges'].append((node1, node2))  # 存储的时候是list，在fcg也更新了nodes和edges
                mutation.mutation['feature_type'] = 'add_edge'
                # node1_upstream = self.check_node_upstream(node1)
                # node2_upstream = self.check_node_upstream(node2)
                #取交集
                # intersection = node1_upstream.intersection(node2_upstream)
                # if len(intersection) != 0:
                #     print("intersection", intersection)
                # mutation.save_log()
                visited.add((node1, node2))
                # print("add edge finished!!!!")
                return mutation

            if len(visited) == len(target1) * len(target2):
                return  None

        return None  # 如果没有添加任何边，返回-1

    def build_mutation_add_edge_type2(self):
        # 找两个节点，node1敏感节点下游，但是node1不是敏感节点，node2没有敏感节点下游

        # print("add edge")
        # user -> system
        mutation = Mutation(self.apk_name)
        visited = set(self.boundary_edges)

        # 补集
        target1 = self.sensitive_user_defined_nodes

        list1 = set(self.system_nodes) - set(self.sensitive_system_nodes)
        list2 = set(self.user_defined_nodes) - set(self.sensitive_user_defined_nodes)
        target2 = list(list1.union(list2))

        if len(target1) == 0 or len(target2) == 0:
            return None


        # label = 'add_edge_type2'
        while True:
            node1 = random.choice(target1)

            node2 = random.choice(target2)

            if (node1, node2) not in visited:
                # print("add edge:", node1, node2)

                # clear mutation
                # mutation.clear_mutation()
                mutation.mutation['add_edges'].append((node1, node2))  # 存储的时候是list，在fcg也更新了nodes和edges
                mutation.mutation['feature_type'] = 'add_edge'
                # node1_upstream = self.check_node_upstream(node1)
                # node2_upstream = self.check_node_upstream(node2)
                # 取交集
                # intersection = node1_upstream.intersection(node2_upstream)
                # if len(intersection) != 0:
                #     print("intersection", intersection)
                # mutation.save_log()
                visited.add((node1, node2))
                # print("add edge finished!!!!")
                return mutation

            if len(visited) == len(target1) * len(target2):
                return None

        return None  # 如果没有添加任何边，返回-1

    def build_mutation_add_edge_type3(self):
        # 找两个节点，node1没有敏感节点下游，node2有敏感节点下游，node2不是敏感节点，保证degree不变

        # print("add edge")
        # user -> system
        mutation = Mutation(self.apk_name)
        visited = set(self.boundary_edges)

        # 补集
        target1 = list(set(self.user_defined_nodes) - set(self.sensitive_user_defined_nodes))

        target2 = self.sensitive_system_nodes
        if len(target2) == 0:
            target2 = self.sensitive_user_defined_nodes

        if len(target1) == 0 or len(target2) == 0:
            return None

        label = 'add_edge_type3'
        while True:
            node1 = random.choice(target1)

            node2 = random.choice(target2)

            if (node1, node2) not in visited:
                # print("add edge:", node1, node2)

                # clear mutation
                # mutation.clear_mutation()
                mutation.mutation['add_edges'].append((node1, node2))  # 存储的时候是list，在fcg也更新了nodes和edges
                mutation.mutation['feature_type'] = 'add_edge'
                # node1_upstream = self.check_node_upstream(node1)
                # node2_upstream = self.check_node_upstream(node2)
                # 取交集
                # intersection = node1_upstream.intersection(node2_upstream)
                # if len(intersection) != 0:
                #     print("intersection", intersection)
                # mutation.save_log()
                visited.add((node1, node2))
                # print("add edge finished!!!!")
                return mutation

            if len(visited) == len(target1) * len(target2):
                return None

        return None  # 如果没有添加任何边，返回-1

    def build_mutation_add_edge_type4(self):
        # 找两个节点，node1没有敏感节点下游，node2是敏感节点

        # print("add edge")
        # user -> system
        mutation = Mutation(self.apk_name)
        visited = set(self.boundary_edges)

        # 补集
        target1 = list(set(self.user_defined_nodes) - set(self.sensitive_user_defined_nodes))

        target2 = self.used_sensitive_nodes

        if len(target1) == 0 or len(target2) == 0:
            return None

        # label = 'add_edge_type4'
        while True:
            node1 = random.choice(target1)

            node2 = random.choice(target2)

            if (node1, node2) not in visited:
                # print("add edge:", node1, node2)

                # clear mutation
                # mutation.clear_mutation()
                mutation.mutation['add_edges'].append((node1, node2))  # 存储的时候是list，在fcg也更新了nodes和edges
                mutation.mutation['feature_type'] = 'add_edge'
                # node1_upstream = self.check_node_upstream(node1)
                # node2_upstream = self.check_node_upstream(node2)
                # 取交集
                # intersection = node1_upstream.intersection(node2_upstream)
                # if len(intersection) != 0:
                #     print("intersection", intersection)
                # mutation.save_log()
                visited.add((node1, node2))
                # print("add edge finished!!!!")
                return mutation

            if len(visited) == len(target1) * len(target2):
                return None

        return None  # 如果没有添加任何边，返回-1

    def _find_caller_callee(self, node):
        callers = [m for m, n in self.edges if n == node and m != n]
        callees = [m for n, m in self.edges if n == node]
        return callers, callees

    def check_downstream(self, node):
        print("check downstream start!!!!")

        # 找到 node1 的所有下游
        downstream = set()
        queue = deque()
        queue.append(node)

        while queue:
            current_node = queue.popleft()
            for neighbor in self.current_call_graph.neighbors(current_node):
                if neighbor not in downstream:
                    downstream.add(neighbor)
                    queue.append(neighbor)

        # 检查找到的下游节点中哪些是敏感节点
        sensitive_downstream = downstream.intersection(set(self._apis))
        return sensitive_downstream

    def init_sensitive_upstream(self):
        start = time.time()
        # print("check upstream start!!!!")
        # 字典用于存储每个敏感节点及其上游节点

        sensitive_edges = set()
        sensitive_nodes = set()
        sensitive_system_nodes = set()
        sensitive_user_defined_nodes = set()

        if self.tree_shap_value is not None:


            # # Step 1: Identify indices of positive and negative values
            # positive_indices = np.where(self.tree_shap_value > 0)[0]
            # negative_indices = np.where(self.tree_shap_value < 0)[0]
            #
            # # Step 2: Sort the values and get sorted indices
            # sorted_pos_indices = positive_indices[np.argsort(self.tree_shap_value[positive_indices])]
            # sorted_neg_indices = negative_indices[np.argsort(self.tree_shap_value[negative_indices])]
            #
            # # sorted_pos_indices and sorted_neg_indices now hold the original indices of the sorted positive and negative values
            # # print("Sorted positive indices:", sorted_pos_indices)
            # # print("Sorted negative indices:", sorted_neg_indices)
            #
            # length_pos = int(len(sorted_pos_indices) * 0.1)
            # length_neg = int(len(sorted_neg_indices) * 0.1)
            #
            # final_sorted_pos_indices = sorted_pos_indices[:length_pos]
            # final_sorted_neg_indices = sorted_neg_indices[:length_neg]


            # print("length_pos", length_pos)
            # print("length_neg", length_neg)
            for i in range(len(self.sensitive_nodes)):
                if self.sensitive_nodes[i] != -1 and self.tree_shap_value[i] != 0:
                # if self.sensitive_nodes[i] != -1 and self.tree_shap_value[i] != 0 and (i in final_sorted_neg_indices or i in final_sorted_pos_indices):
                    node = self.sensitive_nodes[i]
                    # 找到当前敏感节点的所有上游
                    queue = deque()
                    queue.append(node)
                    sensitive_system_nodes.add(node)

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

                            # 记录上游节点之间的边
                            sensitive_edges.add((neighbor, current_node))

        else:

            for i in range(len(self.sensitive_nodes)):
                if self.sensitive_nodes[i] != -1:
                    node = self.sensitive_nodes[i]
                    # 找到当前敏感节点的所有上游
                    queue = deque()
                    queue.append(node)
                    sensitive_system_nodes.add(node)

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

                            # 记录上游节点之间的边
                            sensitive_edges.add((neighbor, current_node))

        # print("all sensitive upstreams", len(sensitive_nodes), len(sensitive_edges))
        self.sensitive_edges = sensitive_edges
        self.sensitive_user_defined_nodes = sensitive_user_defined_nodes
        self.sensitive_system_nodes = sensitive_system_nodes
        end = time.time()
        print("init sensitive upstream finished!!!!", end-start)
        
    def check_node_upstream(self, node):
        # print("check node upstream start!!!!")
        self.generate_new_call_graph()

        upstream = set()
        upstream_edges = set()
        queue = deque()
        queue.append(node)

        while queue:
            current_node = queue.popleft()
            for neighbor in self.current_call_graph.predecessors(current_node):
                if neighbor not in upstream:
                    upstream.add(neighbor)
                    queue.append(neighbor)

                # 记录上游节点之间的边
                upstream_edges.add((neighbor, current_node))
                    
        return upstream, upstream_edges


    def build_mutation_rewiring(self):
        # 1. 随机选择一个边
        # 2. 随机选择一个节点
        # 3. 保证节点不重复
        # 4. 保证边不重复
        mutation = Mutation(self.apk_name)
        visited_edge = set()
        visited_node = set()

        edges = list(self.edges)
        edge_len = len(edges)
        user_defined_nodes = list(self.user_defined_nodes)
        mid_node_len = len(user_defined_nodes)

        if edge_len == 0 or mid_node_len == 0:
            return None

        try_times = 100
        while edge_len > 0 and mid_node_len > 0 and try_times > 0:
            try_times -= 1
            edge_idx = random.randint(0, edge_len-1)

            edge = edges[edge_idx]
            if edge in visited_edge:
                continue
            visited_edge.add(edge)
            begin_node = edge[0]
            end_node = edge[1]
            if begin_node in self.system_nodes:
                continue

            mid_node_idx = random.randint(0, mid_node_len-1)
            mid_node = user_defined_nodes[mid_node_idx]
            if mid_node in visited_node:
                continue
            visited_node.add(mid_node)

            # if begin_node != mid_node and end_node != mid_node:
            mutation.mutation['remove_edges'].append(edge)
            if (begin_node, mid_node) not in self.edges:
                mutation.mutation['add_edges'].append((begin_node, mid_node))

            if (mid_node, end_node) not in self.edges:
                mutation.mutation['add_edges'].append((mid_node, end_node))

            mutation.mutation['feature_type'] = 'rewiring'
            # mutation.save_log()
            return mutation

        return None  # 如果所有边都遍历完且没有符合条件的边，返回 None

    def build_mutation_rewiring_sensitive(self):
        # 1. 随机选择一个边
        # 2. 随机选择一个节点
        # 3. 保证节点不重复
        # 4. 保证边不重复
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
            edge_idx = random.randint(0, edge_len-1)

            edge = edges[edge_idx]
            if edge in visited_edge:
                continue
            visited_edge.add(edge)
            begin_node = edge[0]
            end_node = edge[1]
            if begin_node in self.system_nodes:
                continue

            mid_node_idx = random.randint(0, mid_node_len-1)
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
            # mutation.save_log()
            return mutation

        return None  # 如果所有边都遍历完且没有符合条件的边，返回 None

    def build_mutation_rewiring_type1(self, number = 1):
        # 选一条边，是下游有sensitive的，但是保证终点不是sensitive，保证degree不变，降低敏感度
        mutation = Mutation(self.apk_name)
        visited = set()

        target1 = self.sensitive_edges

        target2 = set(self.user_defined_nodes) - set(self.sensitive_user_defined_nodes)
        target2 = list(target2)

        if len(target1) == 0 or len(target2) == 0:
            return None

        while True:

            while True:
                edge = random.choice(target1)
                if edge not in visited:
                    visited.add(edge)
                    if edge[0] in self.user_defined_nodes and edge[1] not in self.used_sensitive_nodes:
                        break

                if len(visited) == len(target1):
                    return None

            begin_node = edge[0]
            end_node = edge[1]

            #图中可能没有这个点，因为sensitive edges可能出现数据不一致
            if end_node not in self.current_call_graph or begin_node not in self.current_call_graph:
                self.sensitive_edges.remove(edge)
                continue

            if not nx.has_path(self.current_call_graph, end_node, begin_node):
                mid_node_list = random.sample(target2, number)
                mutation.mutation['remove_edges'].append(edge)
                temp_begin_node = begin_node
                for i in range(len(mid_node_list)-1):
                    mid_node = mid_node_list[i]
                    if not nx.has_path(self.current_call_graph, temp_begin_node, mid_node):
                        mutation.mutation['add_edges'].append((temp_begin_node, mid_node))
                    temp_begin_node = mid_node

                if not nx.has_path(self.current_call_graph, temp_begin_node, end_node):
                    mutation.mutation['add_edges'].append((temp_begin_node, end_node))

                mutation.mutation['feature_type'] = 'rewiring'
                # mutation.save_log()
                return mutation
        return None  # 如果所有边都遍历完且没有符合条件的边，返回 None

    def build_mutation_rewiring_type2(self, number = 1):
        # 选一条边，没有sensitive下游,点是有sensitive下游的，
        mutation = Mutation(self.apk_name)
        visited = set()

        target1 = set(self.edges) - set(self.sensitive_edges)
        target1 = list(target1)

        target2 = set(self.sensitive_user_defined_nodes)
        target2 = list(target2)

        if len(target1) == 0 or len(target2) == 0:
            return None

        while True:

            while True:
                edge = random.choice(target1)
                if edge not in visited:
                    visited.add(edge)
                    if edge[0] in self.user_defined_nodes:
                        break

                if len(visited) == len(target1):
                    return None

            begin_node = edge[0]
            end_node = edge[1]

            if not nx.has_path(self.current_call_graph, end_node, begin_node):
                mid_node_list = random.sample(target2, number)
                mutation.mutation['remove_edges'].append(edge)
                temp_begin_node = begin_node
                for i in range(len(mid_node_list) - 1):
                    mid_node = mid_node_list[i]
                    if not nx.has_path(self.current_call_graph, temp_begin_node, mid_node):
                        mutation.mutation['add_edges'].append((temp_begin_node, mid_node))
                    temp_begin_node = mid_node

                if not nx.has_path(self.current_call_graph, temp_begin_node, end_node):
                    mutation.mutation['add_edges'].append((temp_begin_node, end_node))

                mutation.mutation['feature_type'] = 'rewiring'
                # mutation.save_log()
                return mutation
        return None  # 如果所有边都遍历完且没有符合条件的边，返回 None

    def build_mutation_add_node(self):
        #原始的add node方法
        mutation = Mutation(self.apk_name)
        # 1. 随机生成一个节点
        new_node = FCG.number + 40000  # 生成一个新节点
        FCG.number = FCG.number + 1

        user_defined_nodes = list(self.user_defined_nodes)
        node_len = len(user_defined_nodes)
        if node_len == 0:
            return None
        node_idx = random.randint(0, node_len-1)
        node = user_defined_nodes[node_idx]

        # label = None
        # if node in self.sensitive_user_defined_nodes:
        #     label = 'sensitive_user_defined_nodes'
        # else:
        #     label = 'normal_user_defined_nodes'

        mutation.mutation['add_nodes'].append(new_node)
        mutation.mutation['add_edges'].append((node, new_node))
        mutation.mutation['feature_type'] = 'add_node'
        return mutation


    def build_mutation_add_node_type1(self):
        #原始的add node方法
        mutation = Mutation(self.apk_name)
        # 1. 随机生成一个节点
        new_node = "newclass->newmethod_" + str(time.time()) + "_" + str(self.number) + "(II)I;"  # 生成一个新节点
        self.number = self.number + 1

        node = random.choice(self.user_defined_nodes)

        # label = None
        # if node in self.sensitive_user_defined_nodes:
        #     label = 'sensitive_user_defined_nodes'
        # else:
        #     label = 'normal_user_defined_nodes'

        mutation.mutation['add_nodes'].append(new_node)
        mutation.mutation['add_edges'].append((node, new_node))
        mutation.mutation['feature_type'] = 'add_node'
        return mutation

    def build_mutation_insert_node(self):
        #选择一个敏感的边，添加一个节点，连接到敏感边的终点
        mutation = Mutation(self.apk_name)
        # 1. 随机生成一个节点
        new_node = FCG.number + 20000  # 生成一个新节点
        FCG.number = FCG.number + 1

        visited = set()

        try_times = 100
        while try_times > 0:
            edge = random.choice(self.edges)
            if edge in visited:
                try_times -= 1
                continue
            visited.add(edge)
            if edge[0] in self.system_nodes:
                try_times -= 1
                continue

            mutation.mutation['add_nodes'].append(new_node)
            mutation.mutation['remove_edges'].append((edge[0], edge[1]))
            mutation.mutation['add_edges'].append((edge[0], new_node))
            mutation.mutation['add_edges'].append((new_node, edge[1]))
            mutation.mutation['feature_type'] = 'insert_node'
            return mutation

        return None

    def build_mutation_remove_node(self):
        mutation = Mutation(self.apk_name)
        # caller!=0
        # callee!=0


        # print("remove node: len", len(self.user_defined_nodes))

        # print("boundary edges", self.boundary_edges)

        user_defined_nodes = list(self.user_defined_nodes)
        node_len = len(user_defined_nodes)

        try_times = 100
        while node_len > 0 and try_times > 0:
            try_times -= 1


            node_idx = random.randint(0, node_len-1)
            node = user_defined_nodes[node_idx]

            callers = [m for m, n in self.edges if n == node and m != n]  # 有的边是自己指向自己的，包括在calleee中，就不用包括在caller中
            callees = [m for n, m in self.edges if n == node]

            # print("callers", len(callers))
            # print("callees", len(callees))

            # print("remove node: len", len(visited), len(self.user_defined_nodes), len(callers), len(callees))

            # print("callee", callee)
            for caller in callers:
                # print("remove node: remove edge: edges", caller, node)
                mutation.mutation['remove_edges'].append((caller, node))

            for callee in callees:
                # print("remove node: remove edge: edges", node, callee)
                mutation.mutation['remove_edges'].append((node, callee))

            mutation.mutation['remove_nodes'].append(node)
            mutation.mutation['feature_type'] = 'remove_node'

            for caller in callers:
                for callee in callees:
                    if (caller, callee) not in self.edges:
                        mutation.mutation['add_edges'].append((caller, callee))

            return mutation



        return None  # 如果所有节点都遍历完且没有符合条件的节点，返回 None

    def build_mutation_remove_node_sensitive(self):
        mutation = Mutation(self.apk_name)
        # caller!=0
        # callee!=0


        # print("remove node: len", len(self.user_defined_nodes))

        # print("boundary edges", self.boundary_edges)

        sensitive_user_defined_nodes = list(self.sensitive_user_defined_nodes)
        node_len = len(sensitive_user_defined_nodes)

        try_times = 100
        while node_len > 0 and try_times > 0:
            try_times -= 1


            node_idx = random.randint(0, node_len-1)
            node = sensitive_user_defined_nodes[node_idx]

            callers = [m for m, n in self.edges if n == node and m != n]  # 有的边是自己指向自己的，包括在calleee中，就不用包括在caller中
            callees = [m for n, m in self.edges if n == node]

            # print("callers", len(callers))
            # print("callees", len(callees))

            # print("remove node: len", len(visited), len(self.user_defined_nodes), len(callers), len(callees))

            # print("callee", callee)
            for caller in callers:
                # print("remove node: remove edge: edges", caller, node)
                mutation.mutation['remove_edges'].append((caller, node))

            for callee in callees:
                # print("remove node: remove edge: edges", node, callee)
                mutation.mutation['remove_edges'].append((node, callee))

            mutation.mutation['remove_nodes'].append(node)
            mutation.mutation['feature_type'] = 'remove_node'

            for caller in callers:
                for callee in callees:
                    if (caller, callee) not in self.edges:
                        mutation.mutation['add_edges'].append((caller, callee))

            return mutation



        return None  # 如果所有节点都遍历完且没有符合条件的节点，返回 None

    def build_mutation_remove_node_backup(self):
        mutation = Mutation(self.apk_name)
        # caller!=0
        # callee!=0

        # 1. 随机选择一个节点
        visited = set()

        # print("remove node: len", len(self.user_defined_nodes))

        # print("boundary edges", self.boundary_edges)

        target1 = self.sensitive_edges
        if len(target1) == 0:
            return None, None

        while True:

            while True:
                edge = random.choice(target1)
                if edge not in visited:
                    visited.add(edge)
                    if edge[0] in self.sensitive_user_defined_nodes:
                        break
                if len(visited) == len(target1):
                    return None

            node = edge[0]
            callers = [m for m, n in self.edges if n == node and m != n]  # 有的边是自己指向自己的，包括在calleee中，就不用包括在caller中
            callees = [m for n, m in self.edges if n == node]

            # print("callers", len(callers))
            # print("callees", len(callees))

            # print("remove node: len", len(visited), len(self.user_defined_nodes), len(callers), len(callees))

            # print("callee", callee)
            if len(callers) != 0:
                for caller in callers:
                    # print("remove node: remove edge: edges", caller, node)
                    mutation.mutation['remove_edges'].append((caller, node))

                for callee in callees:
                    # print("remove node: remove edge: edges", node, callee)
                    mutation.mutation['remove_edges'].append((node, callee))

                mutation.mutation['remove_nodes'].append(node)
                mutation.mutation['feature_type'] = 'remove_node'

                for caller in callers:
                    for callee in callees:
                        if (caller, callee) not in self.edges:
                            mutation.mutation['add_edges'].append((caller, callee))

                return mutation, None
        return None, None  # 如果所有节点都遍历完且没有符合条件的节点，返回 None

    def build_mutation_remove_node_type1(self):
        #在敏感区域内，选择了一个边
        mutation = Mutation(self.apk_name)
        # caller!=0
        # callee!=0

        # 1. 随机选择一个节点

        # print("remove node: len", len(self.user_defined_nodes))

        # print("boundary edges", self.boundary_edges)
        # for i in range(len(self.sensitive_edges) - 1, -1, -1):  # 从列表末尾开始向前迭代
        #     edge = self.sensitive_edges[i]
        #     if edge[0] not in self.current_call_graph.nodes or edge[1] not in self.current_call_graph.nodes:
        #         del self.sensitive_edges[i]  # 安全地删除元素

        sensitive_edges =  self.sensitive_edges.copy()
        random.shuffle(sensitive_edges)

        for edge in sensitive_edges:
            if edge[0] not in self.user_defined_nodes:
                continue

            node = edge[0]#target node
            callers = [m for m, n in self.current_call_graph.edges if n == node and m != n]  # 有的边是自己指向自己的，包括在calleee中，就不用包括在caller中
            callees = [m for n, m in self.current_call_graph.edges if n == node]


            # print("callers", len(callers))
            # print("callees", len(callees))

            # print("remove node: len", len(visited), len(self.user_defined_nodes), len(callers), len(callees))

            # print("callee", callee)
            if len(callers) != 0:
                for caller in callers:
                    # print("remove node: remove edge: edges", caller, node)
                    if (caller, node) in self.edges:
                        mutation.mutation['remove_edges'].append((caller, node))

                for callee in callees:
                    # print("remove node: remove edge: edges", node, callee)
                    if (node, callee) in self.edges:
                        mutation.mutation['remove_edges'].append((node, callee))

                mutation.mutation['remove_nodes'].append(node)
                mutation.mutation['feature_type'] = 'remove_node'

                for caller in callers:
                    for callee in callees:
                        if (caller, callee) not in self.edges:
                            mutation.mutation['add_edges'].append((caller, callee))

                return mutation
        return None  # 如果所有节点都遍历完且没有符合条件的节点，返回 None

    def build_mutation_remove_node_type2(self):
        # 在敏感区域内，选择了一个边
        mutation = Mutation(self.apk_name)
        # caller!=0
        # callee!=0

        # 1. 随机选择一个节点

        # print("remove node: len", len(self.user_defined_nodes))

        # print("boundary edges", self.boundary_edges)
        # for i in range(len(self.boundary_edges) - 1, -1, -1):  # 从列表末尾开始向前迭代
        #     edge = self.boundary_edges[i]
        #     if edge[0] not in self.current_call_graph.nodes or edge[1] not in self.current_call_graph.nodes:
        #         del self.boundary_edges[i]  # 安全地删除元素

        boundary_edges = self.boundary_edges.copy()
        random.shuffle(boundary_edges)

        for edge in boundary_edges:
            if edge[0] not in self.user_defined_nodes:
                continue

            node = edge[0]  # target node
            callers = [m for m, n in self.current_call_graph.edges if n == node and m != n]  # 有的边是自己指向自己的，包括在calleee中，就不用包括在caller中
            callees = [m for n, m in self.current_call_graph.edges if n == node]

            # print("callers", len(callers))
            # print("callees", len(callees))

            # print("remove node: len", len(visited), len(self.user_defined_nodes), len(callers), len(callees))

            # print("callee", callee)
            if len(callers) != 0:
                for caller in callers:
                    # print("remove node: remove edge: edges", caller, node)
                    if (caller, node) in self.edges:
                        mutation.mutation['remove_edges'].append((caller, node))

                for callee in callees:
                    # print("remove node: remove edge: edges", node, callee)
                    if (node, callee) in self.edges:
                        mutation.mutation['remove_edges'].append((node, callee))

                mutation.mutation['remove_nodes'].append(node)
                mutation.mutation['feature_type'] = 'remove_node'

                for caller in callers:
                    for callee in callees:
                        if (caller, callee) not in self.edges:
                            mutation.mutation['add_edges'].append((caller, callee))

                return mutation
        return None  # 如果所有节点都遍历完且没有符合条件的节点，返回 None

    def build_mutation_add_density_node(self):
        # 原始的add node方法
        mutation = Mutation(self.apk_name)
        # 1. 生成200个节点
        new_nodes = []
        for i in range(200):
            new_node = FCG.number + 40000
            FCG.number = FCG.number + 1
            new_nodes.append(new_node)

        user_defined_nodes = list(self.user_defined_nodes)
        node_len = len(user_defined_nodes)
        if node_len == 0:
            return None
        node_idx = random.randint(0, node_len - 1)
        node = user_defined_nodes[node_idx]
        begin_node = node
        for new_node in new_nodes:
            mutation.mutation['add_nodes'].append(new_node)
            mutation.mutation['add_edges'].append((begin_node, new_node))
            begin_node = new_node

        sensitive_system_nodes = list(self.sensitive_system_nodes)
        sensitive_system_nodes_len = len(sensitive_system_nodes)
        if sensitive_system_nodes_len == 0:
            return None
        sensitive_system_node_idx = random.randint(0, sensitive_system_nodes_len - 1)
        end_node = sensitive_system_nodes[sensitive_system_node_idx]
        mutation.mutation['add_edges'].append((begin_node, end_node))

        mutation.mutation['feature_type'] = 'add_node'

        return mutation

    def build_mutation_merge(self):
        #1.选择1个sensitive nodes的两个节点，保证上游无交集，下游交集合并

        mutation = Mutation(self.apk_name)

        target = set(self.sensitive_user_defined_nodes)
        target = list(target)

        if len(target) == 0:
            return None

        visited = set()

        while True:
            while True:

                nodes = random.sample(target,2)
                if nodes[0] not in visited and nodes[1] not in visited:
                    visited.add(nodes[0])
                    visited.add(nodes[1])
                    break

                if len(visited) == len(target):
                    return None

            upstream_node1 = self.check_node_upstream(nodes[0])
            upstream_node2 = self.check_node_upstream(nodes[1])

        #     if not nx.has_path(self.current_call_graph, end_node, begin_node):
        #         mid_node_list = random.sample(target2, number)
        #         mutation.mutation['remove_edges'].append(edge)
        #         temp_begin_node = begin_node
        #         for i in range(len(mid_node_list) - 1):
        #             mid_node = mid_node_list[i]
        #             if not nx.has_path(self.current_call_graph, temp_begin_node, mid_node):
        #                 mutation.mutation['add_edges'].append((temp_begin_node, mid_node))
        #             temp_begin_node = mid_node
        # 
        #         if not nx.has_path(self.current_call_graph, temp_begin_node, end_node):
        #             mutation.mutation['add_edges'].append((temp_begin_node, end_node))
        # 
        #         mutation.mutation['feature_type'] = 'rewiring'
        #         # mutation.save_log()
        #         return mutation
        # return None  # 如果所有边都遍历完且没有符合条件的边，返回 None

