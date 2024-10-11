import sys
import os
import time
sys.path.append(os.path.abspath('../'))
from Other.ExtractFeature import obtain_sensitive_apis, extract_degree_centrality_from_CG, \
    extract_katz_centrality_from_CG, extract_closeness_centrality_from_CG, extract_harmonic_centrality_from_CG_new, extract_closeness_centrality_from_CG_new
import os
import networkx as nx
import random
from Mutation import Mutation
from collections import deque
import xml.etree.ElementTree as ET


class FCG_malscan:
    number = 0  # The number of new nodes, to avoid conflicts

    def __init__(self, file_path, label, shap_value=None, tree_shap_value=None):
        self.original_call_graph = None  # Initialized once
        self.current_call_graph = None
        self.apk_name = None  # Initialized once

        self.nodes = None  # Updated in real-time
        self.edges = None  # Updated in real-time
        self.user_defined_nodes = None  # Updated in real-time
        self.system_nodes = None  # Updated in real-time
        self.boundary_edges = None  # Updated in real-time, user-->system, no longer used since Jan 21
        self.sensitive_nodes = None  # Updated in real-time, a list of over 20,000 nodes

        # Sensitive reachable areas
        self.sensitive_system_nodes = None  # Updated in real-time
        self.sensitive_user_defined_nodes = None  # Updated in real-time
        self.sensitive_edges = None  # Updated in real-time

        self.used_sensitive_nodes = None  # Initialized once, nodes in sensitive_nodes that are not -1
        self.original_label = None  # Initialized once

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

        self._framework_java = {"java.", "sun.", "android.", "androidx.", "org.apache.", "org.eclipse.", "soot.",
                                "javax.",
                                "com.google.", "org.xml.", "junit.", "org.json.", "org.w3c.dom."}

        self._apis = obtain_sensitive_apis()  # Convert to a set for efficiency in lookup

        self._load(file_path, label)

        self._init_call_graph()

    def _init_call_graph(self):
        self.original_call_graph.copy()

        # 1. Initialize nodes
        nodes = list(self.original_call_graph.nodes._nodes.keys())
        nodes_ids = set()
        user_defined_nodes = set()
        system_nodes = set()
        sensitive_nodes = []

        for i in range(len(nodes)):
            nodes_ids.add(i)
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

        # 2. Initialize edges
        edges = list(self.original_call_graph.edges)
        edges_ids = set()
        boundary_edges = set()

        # Convert edge nodes to node IDs
        for edge in edges:
            u, v = edge
            u_id = node_to_id[u]
            v_id = node_to_id[v]
            edge_id = (u_id, v_id)
            edges_ids.add(edge_id)
            if u_id in self.user_defined_nodes and v_id in self.system_nodes:
                boundary_edges.add(edge_id)

        self.edges = edges_ids

        # Generate the new call graph and initialize sensitive upstream data
        self.generate_new_call_graph()
        self.init_sensitive_upstream()

    def process_mutation(self, mutation):
        mutation = mutation.mutation

        mutation_type = mutation.get('feature_type', '')
        add_edges = mutation.get('add_edges', [])
        remove_edges = mutation.get('remove_edges', [])
        add_nodes = mutation.get('add_nodes', [])
        remove_nodes = mutation.get('remove_nodes', [])

        state = True
        res = {}

        if mutation_type == 'remove_node':
            remove_node = remove_nodes[0]
            if remove_node not in self.nodes and remove_node not in self.user_defined_nodes:
                res["remove_nodes"] = remove_node
                state = False
                return state, res

            flag = False
            if remove_node in self.sensitive_user_defined_nodes:
                self.sensitive_user_defined_nodes.remove(remove_node)
                flag = True

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

        elif mutation_type == 'add_edge':
            add_edge = add_edges[0]
            add_edge = tuple(add_edge)
            if add_edge in self.edges and (add_edge[0] is None or add_edge[1] is None):
                res["add_edges"] = add_edge
                state = False
                return state, res
            else:
                self.edges.add(add_edge)
                if add_edge[1] in self.sensitive_user_defined_nodes or add_edge[1] in self.sensitive_system_nodes:
                    begin_node = add_edge[0]
                    upstream, upstream_edges = self.check_node_upstream(begin_node)
                    self.sensitive_user_defined_nodes.update(upstream)
                    self.sensitive_edges.update(upstream_edges)
                return state, res

        elif mutation_type == 'rewiring':
            remove_edge = remove_edges[0]
            remove_edge = tuple(remove_edge)
            if remove_edge not in self.edges and remove_edge not in self.sensitive_edges:
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

        elif mutation_type == 'add_node':
            if len(add_nodes) == 1 or len(add_edges) == 1:
                add_node = add_nodes[0]
                if add_node in self.nodes:
                    res["add_nodes"] = add_node
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
            print("load the original FCG finished!!!!")
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

    def cal_centralities(self, type='all'):
        """Return the centrality of the current call graph."""
        state = self.generate_new_call_graph()

        if not state:
            return False

        try:
            if type == 'all':
                self.degree_feature = extract_degree_centrality_from_CG(self.current_call_graph, self.sensitive_nodes)
                self.katz_feature = extract_katz_centrality_from_CG(self.current_call_graph, self.sensitive_nodes)
                self.closeness_feature = extract_closeness_centrality_from_CG(self.current_call_graph,
                                                                              self.sensitive_nodes)
                self.harmonic_feature = extract_harmonic_centrality_from_CG_new(self.current_call_graph,
                                                                                self.sensitive_nodes,
                                                                                self.used_sensitive_nodes)
                self.closeness_feature = extract_closeness_centrality_from_CG_new(self.current_call_graph,
                                                                                  self.sensitive_nodes)
                return True

            elif type == 'degree':
                self.degree_feature = extract_degree_centrality_from_CG(self.current_call_graph, self.sensitive_nodes)
                return True

            elif type == 'katz':
                start = time.time()
                self.katz_feature = extract_katz_centrality_from_CG(self.current_call_graph, self.sensitive_nodes)
                return True

            elif type == 'closeness':
                self.closeness_feature = extract_closeness_centrality_from_CG_new(self.current_call_graph,
                                                                                  self.sensitive_nodes)
                return True

            elif type == 'harmonic':
                self.harmonic_feature = extract_harmonic_centrality_from_CG_new(self.current_call_graph,
                                                                                self.sensitive_nodes,
                                                                                self.used_sensitive_nodes)
                return True

        except Exception as e:
            print(f"An error occurred: {e}")
            return False

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



    def _find_caller_callee(self, node):
        callers = [m for m, n in self.edges if n == node and m != n]
        callees = [m for n, m in self.edges if n == node]
        return callers, callees

    def init_sensitive_upstream(self):
        start = time.time()

        # Dictionary to store upstream nodes and edges for each sensitive node
        sensitive_edges = set()
        sensitive_nodes = set()
        sensitive_system_nodes = set()
        sensitive_user_defined_nodes = set()

        if self.tree_shap_value is not None:
            for i in range(len(self.sensitive_nodes)):
                if self.sensitive_nodes[i] != -1 and self.tree_shap_value[i] != 0:
                    node = self.sensitive_nodes[i]
                    # Find all upstream nodes for the current sensitive node
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

                            # Record edges between upstream nodes
                            sensitive_edges.add((neighbor, current_node))

        else:
            for i in range(len(self.sensitive_nodes)):
                if self.sensitive_nodes[i] != -1:
                    node = self.sensitive_nodes[i]
                    # Find all upstream nodes for the current sensitive node
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

                            # Record edges between upstream nodes
                            sensitive_edges.add((neighbor, current_node))

        self.sensitive_edges = sensitive_edges
        self.sensitive_user_defined_nodes = sensitive_user_defined_nodes
        self.sensitive_system_nodes = sensitive_system_nodes
        end = time.time()
        # print("init sensitive upstream finished!!!!", end - start)
        
    def check_node_upstream(self, node):
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


                upstream_edges.add((neighbor, current_node))
                    
        return upstream, upstream_edges

    def build_mutation_rewiring(self):
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

    def build_mutation_add_node(self):
        mutation = Mutation(self.apk_name)
        new_node = self.number + 40000
        self.number = self.number + 1

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
