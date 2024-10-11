#每个是一个组
#MutationSequence	fcg	保存原图
# mutation_sequence	Mutation的list的list，要保证mutation_sequence[i]都是可分离的
# truncate	截取
# concatenate	拼接
# mutation	变异
# load	读取pkl文件
# save	写到pkl文件
import time

from FCG import FCG
from Mutation import Mutation
import random
import json
import os
import copy

class MutationSequence:

    def __init__(self, mutations, state = False):
        self.mutations = mutations
        self.group_list = [] #存储mutation的index, 一个group是一个list，最终有多少组
        self.final_group_list = [] #存储mutation对象
        self.dict_addNode_to_mutationIndex = {}  # 标记新加的点和mutation的关系
        self.dict_addEdge_to_mutationIndex = {}  # 标记新加的边和mutation的关系
        self.dict_removeEdge_to_mutationIndex = {}  # 标记删除的边和mutation的关系
        self.dict_mutationIndex_to_group = {}  # 标记mutation和group的关系

        if not state:
            # when state==false
            # has dependency between mutations
            self.generate_group_list()

        else:
            #ignore the dependency between mutations
            self.final_group_list = [[mutation] for mutation in mutations]


        # self.save_mutations()
        # self.final_group_list = []
        # for mutation in mutations:
        #     self.final_group_list.append([mutation])




    def generate_group_list(self): #初始化

        # new version
        for i in range(len(self.mutations)):
            mutation = self.mutations[i] #是一个mutation对象
            # print("mutation", mutation)
            mutation = mutation.mutation
            current_type = mutation.get('feature_type', '')
            # print(mutation)
            if current_type == 'add_node':
                parent_index = self._find_parent_add_node(mutation)
                # print("parent_index", parent_index)

                if parent_index != -1:
                    # 找到parent对应的group，把当前的mutation加入到group中
                    group = self.dict_mutationIndex_to_group[parent_index]
                    group.append(i)
                    self.dict_mutationIndex_to_group[i] = group
                else:
                    group = [i]
                    self.group_list.append(group)
                    self.dict_mutationIndex_to_group[i] = group

                data = mutation.get('add_nodes', [])[0]
                data = json.dumps(data)
                self.dict_addNode_to_mutationIndex[data] = i
                data = mutation.get('add_edges', [])[0]
                data = json.dumps(data)
                self.dict_addEdge_to_mutationIndex[data] = i

            elif current_type == 'add_edge':
                # 可能有多个parent
                parents_indexes = self._find_parent_add_edge(mutation)
                # print("parent_index", parents_indexes)

                if len(parents_indexes) != 0:
                    if len(parents_indexes) == 1:
                        # 找到parent对应的group，把当前的mutation加入到group中
                        group = self.dict_mutationIndex_to_group[parents_indexes[0]]
                        group.append(i)
                        self.dict_mutationIndex_to_group[i] = group
                    else:
                        # 有多个父节点，需要合并
                        # 全都合并到第一个父节点的group中
                        groups = self.merge_parent_group(parents_indexes)
                        merged_group = groups[0]
                        # print("merged_group")
                        for group in groups[1:]:
                            # print("group", len(group))
                            for mutation_id in group:
                                # 每一个mutation都加入到第一个父节点的group中，并更新dict_node_to_group
                                merged_group.append(mutation_id)
                                self.dict_mutationIndex_to_group[mutation_id] = merged_group
                            # 从grouplist中删除group
                            self.group_list.remove(group)
                        # 对于当前的mutation，也加入到第一个父节点的group中
                        merged_group.append(i)
                        self.dict_mutationIndex_to_group[i] = merged_group
                else:
                    group = [i]
                    self.group_list.append(group)
                    self.dict_mutationIndex_to_group[i] = group

                add_edge = mutation.get('add_edges', [])[0]
                add_edge = json.dumps(add_edge)
                self.dict_addEdge_to_mutationIndex[add_edge] = i  # add edge只有一个

            elif current_type == 'rewiring':
                # 可能有多个parent
                parents_indexes = self._find_parent_rewiring(mutation)
                # print("parent_index", parents_indexes)

                if len(parents_indexes) != 0:
                    if len(parents_indexes) == 1:
                        # 找到parent对应的group，把当前的mutation加入到group中
                        group = self.dict_mutationIndex_to_group[parents_indexes[0]]
                        group.append(i)
                        self.dict_mutationIndex_to_group[i] = group
                    else:
                        # 有多个父节点，需要合并
                        # 全都合并到第一个父节点的group中
                        groups = self.merge_parent_group(parents_indexes)
                        merged_group = groups[0]
                        # print("merged_group")
                        for group in groups[1:]:
                            # print("group", len(group))
                            for mutation_id in group:
                                # 每一个mutation都加入到第一个父节点的group中，并更新dict_node_to_group
                                merged_group.append(mutation_id)
                                self.dict_mutationIndex_to_group[mutation_id] = merged_group
                            # 从grouplist中删除group
                            self.group_list.remove(group)
                        # 对于当前的mutation，也加入到第一个父节点的group中
                        merged_group.append(i)
                        self.dict_mutationIndex_to_group[i] = merged_group
                else:
                    group = [i]
                    self.group_list.append(group)
                    self.dict_mutationIndex_to_group[i] = group

                remove_edge = mutation.get('remove_edges', [])[0]
                remove_edge = json.dumps(remove_edge)
                self.dict_removeEdge_to_mutationIndex[remove_edge] = i  # add edge只有一个
                add_edges = mutation.get('add_edges', []) #不确定有几条边
                for add_edge in add_edges:
                    add_edge = json.dumps(add_edge)
                    self.dict_addEdge_to_mutationIndex[add_edge] = i

            elif current_type == 'remove_node':
                # 可能有多个parent
                parents_indexes = self._find_parent_remove_node(mutation)
                # print("parent_index", parents_indexes)

                if len(parents_indexes) != 0:
                    if len(parents_indexes) == 1:
                        # 找到parent对应的group，把当前的mutation加入到group中
                        group = self.dict_mutationIndex_to_group[parents_indexes[0]]
                        group.append(i)
                        self.dict_mutationIndex_to_group[i] = group
                    else:
                        # 有多个父节点，需要合并
                        # 全都合并到第一个父节点的group中
                        groups = self.merge_parent_group(parents_indexes)
                        merged_group = groups[0]
                        # print("merged_group")
                        for group in groups[1:]:
                            # print("group", len(group))
                            for mutation_id in group:
                                # 每一个mutation都加入到第一个父节点的group中，并更新dict_node_to_group
                                merged_group.append(mutation_id)
                                self.dict_mutationIndex_to_group[mutation_id] = merged_group
                            # 从grouplist中删除group
                            self.group_list.remove(group)
                        # 对于当前的mutation，也加入到第一个父节点的group中
                        merged_group.append(i)
                        self.dict_mutationIndex_to_group[i] = merged_group
                else:
                    group = [i]
                    self.group_list.append(group)
                    self.dict_mutationIndex_to_group[i] = group

                remove_edges = mutation.get('remove_edges', [])
                for remove_edge in remove_edges:
                    remove_edge = json.dumps(remove_edge)
                    self.dict_removeEdge_to_mutationIndex[remove_edge] = i

                add_edges = mutation.get('add_edges', [])
                for add_edge in add_edges:
                    add_edge = json.dumps(add_edge)
                    self.dict_addEdge_to_mutationIndex[add_edge] = i

        self.final_group_list = self.translate_group_list()


    def merge_parent_group(self, parent_indexes):
        seen = {}
        groups = []

        for parent_index in parent_indexes:
            group = self.dict_mutationIndex_to_group[parent_index]
            group_tuple = tuple(group)

            if group_tuple not in seen:
                seen[group_tuple] = True
                groups.append(group)

        # print('group_ans', groups)
        return groups


    def print_group_list(self):
        # 遍历group_list，打印依赖关系
        print("-------")
        for i in range(len(self.group_list)):
            print(self.group_list[i])

    def get_mutation_to_group(self):
        for key in self.dict_mutationIndex_to_group:
            print(key, self.dict_mutationIndex_to_group[key])
        print("-----------")
        for key in self.dict_addNode_to_mutationIndex:
            print(key, self.dict_addNode_to_mutationIndex[key])
        print("-----------")
        for key in self.dict_addEdge_to_mutationIndex:
            print(key, self.dict_addEdge_to_mutationIndex[key])
        print("-----------")
        for key in self.dict_removeEdge_to_mutationIndex:
            print(key, self.dict_removeEdge_to_mutationIndex[key])



    def get_one_list_without_repeat(self, one_list):
        # 检查group_list中是否有重复的元素，并返回一个去重后的列表
        seen = set()
        new_list = []
        for item in one_list:
            # 使用字典的字符串表示作为唯一标识
            representation = str(item)
            if representation not in seen:
                seen.add(representation)
                new_list.append(item)
        return new_list

    def _find_parent_add_node(self, current):
        # print("_find_parent_add_node")
        #check:
        #1.begin node是一个新加的点
        add_edge = current.get('add_edges', [])[0]
        begin_node = add_edge[0]
        #如果没有这个key==begin_node，就返回-1
        begin_node = json.dumps(begin_node)
        index = self.dict_addNode_to_mutationIndex.get(begin_node, -1)
        return index

    def _find_parent_add_edge(self, current):
        # print("_find_parent_add_edge")
        # check:
        #1. begin node是一个新加的点
        #2. end node是一个新加的点
        #2. edge 是一个之前删除a的边
        parents_indexes = set()

        add_edge = current.get('add_edges', [])[0]
        #序列化add_edge
        # add_edge = json.dumps(add_edge)
        # index = self.dict_removeEdge_to_mutationIndex.get(add_edge, -1)
        # if index != -1:
        #     parents_indexes.add(index)

        begin_node = add_edge[0]
        begin_node = json.dumps(begin_node)
        index = self.dict_addNode_to_mutationIndex.get(begin_node, -1)
        if index != -1:
            parents_indexes.add(index)

        end_node = add_edge[1]
        end_node = json.dumps(end_node)
        index = self.dict_addNode_to_mutationIndex.get(end_node, -1)
        if index != -1:
            parents_indexes.add(index)

        return list(parents_indexes)

    def _find_parent_rewiring(self, current):
        # print("_find_parent_rewiring")
        #jan 10: dont check the add edges

        parents_indexes = set()

        remove_edge = current.get('remove_edges', [])[0]
        add_edges = current.get('add_edges', []) #不确定有几条边（0，1，2）
        begin_node = remove_edge[0]
        end_node = remove_edge[1]
        mid_node = ''
        if len(add_edges) == 1:
            if add_edges[0][0] == begin_node:
                mid_node = add_edges[0][1]
            elif add_edges[0][1] == end_node:
                mid_node = add_edges[0][0]
        elif len(add_edges) == 2:
            mid_node = add_edges[0][1]


        # print("mid_node", mid_node)
        # print("begin_node", begin_node)
        # print("end_node", end_node)
        # print("len add_edges", len(add_edges))


        # check:
        # 3.remove edge 是一个新加的边
        remove_edge = json.dumps(remove_edge)
        index = self.dict_addEdge_to_mutationIndex.get(remove_edge, -1)
        if index != -1:
            parents_indexes.add(index)


        # 1.begin node是一个新加的点
        # 2.mid node是一个新加的点
        # begin_node = json.dumps(begin_node)
        # index = self.dict_addNode_to_mutationIndex.get(begin_node, -1)
        # if index != -1:
        #     parents_indexes.add(index)

        if mid_node != '':
            mid_node = json.dumps(mid_node)
            index = self.dict_addNode_to_mutationIndex.get(mid_node, -1)
            if index != -1:
                parents_indexes.add(index)

        # 4.end node是一个新加的点
        # end_node = json.dumps(end_node)
        # index = self.dict_addNode_to_mutationIndex.get(end_node, -1)
        # if index != -1:
        #     parents_indexes.add(index)



        #4.add edge 是一个新加的边
        # for add_edge in add_edges:
        #     add_edge = json.dumps(add_edge)
        #     index = self.dict_removeEdge_to_mutationIndex.get(add_edge, -1)
        #     if index != -1:
        #         parents_indexes.add(index)

        return list(parents_indexes)

    def _find_parent_remove_node(self, current):
        # print("_find_parent_remove_node")

        remove_edges = current.get('remove_edges', [])
        add_edges = current.get('add_edges', [])  # 不确定有几条边（0，1，2）
        mid_node = current.get('remove_nodes', [])[0]

        parents_indexes = set()
        # 1.remove edge 是一个新加的边
        # for remove_edge in remove_edges:
        #     remove_edge = json.dumps(remove_edge)
        #     index = self.dict_addEdge_to_mutationIndex.get(remove_edge, -1)
        #     if index != -1:
        #         parents_indexes.add(index)
        #
        # # 2.add edge 是之前删的，跟remove edge有关，关联rewiring
        # for add_edge in add_edges:
        #     add_edge = json.dumps(add_edge)
        #     index = self.dict_removeEdge_to_mutationIndex.get(add_edge, -1)
        #     if index != -1:
        #         parents_indexes.add(index)


        # 3.mid node是一个新加的点
        mid_node = json.dumps(mid_node)
        index = self.dict_addNode_to_mutationIndex.get(mid_node, -1)
        if index != -1:
            parents_indexes.add(index)

        return list(parents_indexes)

    def crossover(self, other_mutation_sequence, fcg):
        
        mutation_list1 = self.final_group_list
        mutation_list2 = other_mutation_sequence.final_group_list

        new_mutation_list = []
        min_length = min(len(mutation_list1), len(mutation_list2))

        # error_mutations = []

        start = time.time()

        cnt_1 = 0
        cnt_0 = 0
        for i in range(min_length):
            choose = random.randint(0, 1)
            if choose == 0:
                # state = fcg.process_mutation_sequence(mutation_list1[i])
                # if state:
                new_mutation_list.append(mutation_list1[i])
                cnt_0 += 1
                # else:
                #     error_mutations.append(mutation_list1[i])
            else:
                # state = fcg.process_mutation_sequence(mutation_list2[i])
                # if state:
                new_mutation_list.append(mutation_list2[i])
                cnt_1 += 1
                # else:
                #     error_mutations.append(mutation_list2[i])

            #一个补偿机制，倒序删除，防止报错
            # indices_to_remove = []
            # # 首先正序遍历列表
            # for j in range(len(error_mutations)):
            #     state = fcg.process_mutation_sequence(error_mutations[j])
            #     if state:
            #         new_mutation_list.append(error_mutations[j])
            #         # 不直接删除，而是记录下需要删除的索引
            #         indices_to_remove.append(j)
            #
            # # 逆序删除元素，以避免索引错位
            # for index in reversed(indices_to_remove):
            #     error_mutations.pop(index)

        # if min_length < len(mutation_list1):
        #     for i in range(min_length, len(mutation_list1)):
        #         state = fcg.process_mutation_sequence(mutation_list1[i])
        #         if state:
        #             new_mutation_list.append(mutation_list1[i])
        #             cnt_0 += 1
        #         # else:
        #         #     error_mutations.append(mutation_list1[i])
        # else:
        #     for i in range(min_length, len(mutation_list2)):
        #         state = fcg.process_mutation_sequence(mutation_list2[i])
        #         if state:
        #             new_mutation_list.append(mutation_list2[i])
        #             cnt_1 += 1
                # else:
                #     error_mutations.append(mutation_list2[i])

        # indices_to_remove = []
        # # 首先正序遍历列表
        # for j in range(len(error_mutations)):
        #     state = fcg.process_mutation_sequence(error_mutations[j])
        #     if state:
        #         new_mutation_list.append(error_mutations[j])
        #         # 不直接删除，而是记录下需要删除的索引
        #         indices_to_remove.append(j)
        #
        # # 逆序删除元素，以避免索引错位
        # for index in reversed(indices_to_remove):
        #     error_mutations.pop(index)


        # for i in range(len(error_mutations)):
        #     state = fcg.process_mutation_sequence(error_mutations[i])
        #     if state:
        #         new_mutation_list.append(error_mutations[i])
        #         error_mutations.remove(error_mutations[i])


        #写入文件
        # res = "crossover cnt_0" + str(cnt_0) + "\n"
        # res += "crossover cnt_1" + str(cnt_1) + "\n"
        # res += "new_mutation_list" + str(len(new_mutation_list)) + "\n"
        # # res += "crossover error_mutations" + str(len(error_mutations)) + "\n"
        #
        # with open("/data/c/shiwensong/malwareGA/task/crossover_jan10.txt", "a") as f:
        #     f.write(res)
        #
        # f.close()

        # print("crossover cnt_0", cnt_0)
        # print("crossover cnt_1", cnt_1)
        # print("crossover error_mutations", len(error_mutations))

        self.final_group_list = new_mutation_list

    def solve_conflict(self, fcg, mutation, res):
        #1.mutation
        type = mutation.mutation.get('feature_type', '')
        res_type = next(iter(res))
        res_place = res[res_type]

        # print("original mutation", mutation.mutation)

        if type == 'add_node':
            if res_type == 'add_nodes' or res_type == 'add_edges':
                #change a new benign node
                begin_node = mutation.mutation.get('add_edges', [])[0][0]
                end_node = mutation.mutation.get('add_edges', [])[0][1]
                nodes = list(fcg.user_defined_nodes)
                nodes_length = len(nodes)
                try_times = 30
                visited = set()
                new_begin_node = None
                while try_times > 0:
                    new_node_id = random.randint(0, nodes_length-1)
                    if new_node_id in visited:
                        continue
                    try_times -= 1
                    visited.add(new_node_id)
                    new_begin_node = nodes[new_node_id]
                    if new_begin_node == begin_node:
                        continue
                    if (new_begin_node, end_node) not in fcg.edges:
                        break
                if (new_begin_node, end_node) not in fcg.edges:
                    mutation.mutation['add_edges'] = [(new_begin_node, end_node)]
                    # print("new mutation", mutation.mutation)
                    fcg.edges.add((new_begin_node, end_node))
                    return True
                else:
                    return False
        elif type == 'add_edge':
            if res_type == 'add_edges':
                edges = fcg.edges
                user_defined_nodes = list(fcg.user_defined_nodes)
                nodes_length = len(user_defined_nodes)
                try_times = 20
                visited = set()
                end_node = mutation.mutation.get('add_edges', [])[0][1]
                #1. change a new benign node
                new_begin_node = None
                while try_times > 0:
                    new_node_id = random.randint(0, nodes_length - 1)
                    if new_node_id in visited:
                        continue
                    try_times -= 1
                    visited.add(new_node_id)
                    new_benign_node = user_defined_nodes[new_node_id]
                    if (new_begin_node, end_node) not in edges:
                        break

                if new_begin_node is not None:
                    mutation.mutation['add_edges'] = [(new_begin_node, end_node)]
                    # print("new mutation", mutation.mutation)
                    fcg.edges.add((new_begin_node, end_node))
                    return True
                else:
                    #2. change a new end node
                    try_times = 20
                    visited = set()
                    benign_node = mutation.mutation.get('add_edges', [])[0][0]
                    nodes = list(fcg.nodes)
                    nodes_length = len(nodes)
                    # 1. change a new benign node
                    new_end_node = None
                    while try_times > 0:
                        new_node_id = random.randint(0, nodes_length - 1)
                        if new_node_id in visited:
                            continue
                        try_times -= 1
                        visited.add(new_node_id)
                        new_end_node = nodes[new_node_id]
                        if (benign_node, new_end_node) not in edges:
                            break
                    if new_end_node is not None:
                        mutation.mutation['add_edges'] = [(benign_node, new_end_node)]
                        # print("new mutation", mutation.mutation)
                        fcg.edges.add((benign_node, new_end_node))
                        return True
                    else:
                        return False
        elif type == 'rewiring':
            if res_type == 'remove_edges':
                edges = fcg.edges
                begin_node = mutation.mutation.get('remove_edges', [])[0][0]
                end_node = mutation.mutation.get('remove_edges', [])[0][1]
                add_edges = mutation.mutation.get('add_edges', [])
                mid_node = ''
                if len(add_edges) == 1:
                    if add_edges[0][0] == begin_node:
                        mid_node = add_edges[0][1]
                    elif add_edges[0][1] == end_node:
                        mid_node = add_edges[0][0]
                elif len(add_edges) == 2:
                    mid_node = add_edges[0][1]

                if mid_node == '':
                    return False

                #1. change a new benign node
                # through finding the caller of end_node
                callers = [m for m, n in edges if n == end_node and m != n]
                callers_length = len(callers)
                print("callers", callers_length)
                try_times = 20
                visited = set()
                new_benign_node = None
                while callers_length > 0 and try_times > 0:
                    new_edge_idx = random.randint(0, callers_length - 1)
                    if len(visited) == callers_length:
                        break
                    if new_edge_idx in visited:
                        continue
                    visited.add(new_edge_idx)
                    try_times -= 1
                    new_benign_node = callers[new_edge_idx]
                    if new_benign_node == mid_node:
                        continue
                    if (new_benign_node, end_node) in edges:
                        break

                if new_benign_node is not None:
                    #process new mutation
                    mutation.mutation['remove_edges'] = [(new_benign_node, end_node)]
                    fcg.edges.remove((new_benign_node, end_node))
                    if len(add_edges) == 1:
                        if add_edges[0][0] == begin_node:
                            mid_node = add_edges[0][1]
                            mutation.mutation['add_edges'] = [(new_benign_node, mid_node)]
                            fcg.edges.add((new_benign_node, mid_node))
                    elif len(add_edges) == 2:
                        mid_node = add_edges[0][1]
                        mutation.mutation['add_edges'] = [(new_benign_node, mid_node), (mid_node, end_node)]
                        fcg.edges.add((new_benign_node, mid_node))
                        fcg.edges.add((mid_node, end_node))

                    # print("new mutation", mutation.mutation)

                    return True
                else:
                    #2. change the end node
                    # through finding the callee of begin_node
                    callees = [m for n, m in fcg.edges if n == begin_node and m != n]
                    callees_length = len(callees)
                    # print("callees", callees_length)
                    try_times = 20
                    visited = set()
                    new_end_node = None
                    while callees_length > 0 and try_times > 0:
                        new_edge_idx = random.randint(0, callees_length - 1)
                        if len(visited) == callees_length:
                            break
                        if new_edge_idx in visited:
                            continue
                        visited.add(new_edge_idx)
                        try_times -= 1
                        new_end_node = callees[new_edge_idx]
                        if (begin_node, new_end_node) in edges:
                            break
                    if new_end_node is not None:
                        mutation.mutation['remove_edges'] = [(begin_node, new_end_node)]
                        fcg.edges.remove((begin_node, new_end_node))
                        add_edges = mutation.mutation.get('add_edges', [])
                        mid_node = ''
                        if len(add_edges) == 1:
                            if add_edges[0][1] == end_node:
                                mid_node = add_edges[0][0]
                                mutation.mutation['add_edges'] = [(mid_node, new_end_node)]
                                fcg.edges.add((mid_node, new_end_node))
                        elif len(add_edges) == 2:
                            mid_node = add_edges[0][1]
                            mutation.mutation['add_edges'] = [(begin_node, mid_node), (mid_node, new_end_node)]
                            fcg.edges.add((begin_node, mid_node))
                            fcg.edges.add((mid_node, new_end_node))

                        print("new mutation", mutation.mutation)
                        return True
                    else:
                        return False

        elif type == 'remove_node':
            return False

        return False
            # if res_type == 'remove_nodes':
            #     visited = set()
            #     user_defined_nodes = list(fcg.user_defined_nodes)
            #     nodes_length = len(user_defined_nodes)
            #     try_times = 20
            #     new_node = None
            #     remove_node = mutation.mutation.get('remove_nodes', [])[0]
            #     add_edges = mutation.mutation.get('add_edges', [])
            #     node_callers = set()
            #     node_callees = set()
            #     can_choose = set()
            #     for add_edge in add_edges:
            #         if add_edge[1] == remove_node:
            #             node_callers.add(add_edge[0])
            #         elif add_edge[0] == remove_node:
            #             node_callees.add(add_edge[1])
            #
            #     for caller in callers:
            #        cur_callees = [m for n, m in fcg.edges if n == caller and m != n]
            #
            #     #find a new node, has insection with callers and callees
            #
            #     while try_times > 0:
            #         new_node_id = random.randint(0, nodes_length - 1)
            #         if new_node_id in visited:
            #             continue
            #         try_times -= 1
            #         visited.add(new_node_id)
            #         new_node = user_defined_nodes[new_node_id]
            #         if new_node != res_place:
            #             break


    #mutation变异的两种方式
    def crossover_single(self, other_mutation_sequence, fcg):
        fcg = copy.deepcopy(fcg)

        mutation_list1 = self.final_group_list
        mutation_list2 = other_mutation_sequence.final_group_list

        new_mutation_list = []
        min_length = min(len(mutation_list1), len(mutation_list2))

        #1.random choose a location
        idx = random.randint(0, min_length)

        #2. concentrate two list
        left_list1, right_list1 = self._truncate(idx)
        left_list2, right_list2 = other_mutation_sequence._truncate(idx)

        #3. concatenate
        new_mutation_list.extend(left_list1)
        new_mutation_list.extend(right_list2)


        self.final_group_list = new_mutation_list

    def _truncate(self, index):
        #截取
        #index是截取比例
        # 确保 index 在列表长度范围内
        mutation_list = self.translate_group_list()

        index = min(index, len(mutation_list) - 1)

        # 截取列表
        mutation_list_left = mutation_list[:index + 1]  # 包含索引位置的元素
        mutation_list_right = mutation_list[index + 1:]  # 从索引的下一个位置开始到列表末尾

        return mutation_list_left, mutation_list_right

    def translate_group_list(self):
        mutation_list = []
        group_list = self.group_list
        # index对应的mutation
        for i in range(len(group_list)):
            group_mutation = []
            # 处理一个group
            for j in range(len(group_list[i])):
                index = group_list[i][j]
                group_mutation.append(self.mutations[index])
            mutation_list.append(group_mutation)

        return mutation_list

    def _concatenate(self, mutation_list1, mutation_list2):
        #拼接
        new_mutation_list = []
        new_mutation_list.extend(mutation_list1)
        new_mutation_list.extend(mutation_list2)
        return new_mutation_list

    def ga_mutation(self, fcg, mutation_pro = 0.2, sensitive = True):
        #fcg: current fcg

        # print("ga_mutation ing")
        group_list_length = len(self.final_group_list)
        # print("group_list", group_list_length)
        # new_group_list = copy.deepcopy(self.final_group_list)#check if it's necessary

        for i in range(group_list_length):
            group = self.final_group_list[i]
            group_length = len(group)
            if group_length == 0:
                continue

            # print("before mutation", len(group))
            pros = random.random()
            if pros < mutation_pro:
                random_operation = random.randint(1, 3)
                if random_operation == 1:# replace
                    location = 0
                    if group_length > 1:
                        location = random.randint(0, group_length - 1)
                    #1.rollback
                    old_mutation = group[location]
                    fcg.rollback_mutation(old_mutation)
                    #2. generate new mutation
                    type_random = {
                        1: 'add_node',
                        2: 'remove_node',
                        3: 'rewiring',
                        4: 'add_edge',
                    }
                    #build a new mutation
                    try_times = 20
                    while try_times > 0:
                        random_int = random.randint(1, 4)
                        # print("random_int", random_int)
                        try_times -= 1

                        if type_random[random_int] != old_mutation.mutation['feature_type']:  # 随机数与当前的不同
                            target_type = type_random[random_int]

                            new_mutation = None
                            if target_type == 'add_node':
                                new_mutation = fcg.build_mutation_add_node()

                            elif target_type == 'add_edge':
                                if sensitive:
                                    new_mutation = fcg.build_mutation_add_edge_sensitive()
                                else:
                                    new_mutation = fcg.build_mutation_add_edge()

                            elif target_type == 'remove_node':
                                if sensitive:
                                    new_mutation = fcg.build_mutation_remove_node_sensitive()
                                else:
                                    new_mutation = fcg.build_mutation_remove_node()

                            elif target_type == 'rewiring':
                                if sensitive:
                                    new_mutation = fcg.build_mutation_rewiring_sensitive()
                                else:
                                    new_mutation = fcg.build_mutation_rewiring()

                            if new_mutation is not None:
                                state = fcg.process_mutation(new_mutation)
                                if state:
                                    #delete the old mutation
                                    self.final_group_list[i].pop(location)
                                    #add the new mutation
                                    self.final_group_list.append([new_mutation])
                                    break
                    #save log
                    # res = "operation: replace; try_times: " + str(try_times) + "\n"
                    # with open("ga_mutation_feb4.txt", "a") as f:
                    #     f.write(res)

                elif random_operation == 2:# delete
                    location = 0
                    if group_length > 1:
                        location = random.randint(0, group_length - 1)
                    # 1.rollback
                    old_mutation = group[location]
                    fcg.rollback_mutation(old_mutation)
                    # 2. delete the old mutation
                    self.final_group_list[i].pop(location)

                elif random_operation == 3:# add
                    # 1. generate new mutation
                    type_random = {
                        1: 'add_node',
                        2: 'remove_node',
                        3: 'rewiring',
                        4: 'add_edge',
                    }
                    # build a new mutation
                    try_times = 20
                    while try_times > 0:
                        random_int = random.randint(1, 4)
                        # print("random_int", random_int)
                        try_times -= 1

                        new_mutation = None
                        if random_int == 1:
                            new_mutation = fcg.build_mutation_add_node()

                        elif random_int == 2:
                            if sensitive:
                                new_mutation = fcg.build_mutation_add_edge_sensitive()
                            else:
                                new_mutation = fcg.build_mutation_add_edge()

                        elif random_int == 3:
                            if sensitive:
                                new_mutation = fcg.build_mutation_remove_node_sensitive()
                            else:
                                new_mutation = fcg.build_mutation_remove_node()

                        elif random_int == 4:
                            if sensitive:
                                new_mutation = fcg.build_mutation_rewiring_sensitive()
                            else:
                                new_mutation = fcg.build_mutation_rewiring()

                        if new_mutation is not None:
                            state = fcg.process_mutation(new_mutation)
                            if state:
                                # add the new mutation
                                self.final_group_list.append([new_mutation])
                                break

                    # save log
                    # res = "operation: add; try_times: " + str(try_times) + "\n"
                    # with open("ga_mutation_jan15_pop200.txt", "a") as f:
                    #     f.write(res)


            # print("after mutation", len(group))

    def ga_mutation_add_edge(self, fcg, mutation_pro = 0.2, sensitive = True):
        #fcg: current fcg

        print("ga_mutation ing")
        group_list_length = len(self.final_group_list)
        # print("group_list", group_list_length)
        # new_group_list = copy.deepcopy(self.final_group_list)#check if it's necessary

        for i in range(group_list_length):
            group = self.final_group_list[i]
            group_length = len(group)
            if group_length == 0:
                continue

            # print("before mutation", len(group))
            pros = random.random()
            if pros < mutation_pro:
                random_operation = random.randint(1, 3)
                if random_operation == 1:# replace
                    location = 0
                    if group_length > 1:
                        location = random.randint(0, group_length - 1)
                    #1.rollback
                    old_mutation = group[location]
                    fcg.rollback_mutation(old_mutation)
                    #2. generate new mutation
                    type_random = {
                        1: 'add_edge'
                    }
                    #build a new mutation
                    try_times = 20
                    while try_times > 0:
                        random_int = 1
                        # print("random_int", random_int)
                        try_times -= 1
                        target_type = type_random[random_int]
                        new_mutation = None
                        if target_type == 'add_edge':
                            if sensitive:
                                new_mutation = fcg.build_mutation_add_edge_sensitive()
                            else:
                                new_mutation = fcg.build_mutation_add_edge()

                        if new_mutation is not None:
                            state = fcg.process_mutation(new_mutation)
                            if state:
                                #delete the old mutation
                                self.final_group_list[i].pop(location)
                                #add the new mutation
                                self.final_group_list.append([new_mutation])
                                break
                    #save log
                    # res = "operation: replace; try_times: " + str(try_times) + "\n"
                    # with open("ga_mutation_feb4.txt", "a") as f:
                    #     f.write(res)

                elif random_operation == 2:# delete
                    location = 0
                    if group_length > 1:
                        location = random.randint(0, group_length - 1)
                    # 1.rollback
                    old_mutation = group[location]
                    fcg.rollback_mutation(old_mutation)
                    # 2. delete the old mutation
                    self.final_group_list[i].pop(location)

                elif random_operation == 3:# add
                    # build a new mutation
                    try_times = 20
                    while try_times > 0:
                        random_int = 1
                        # print("random_int", random_int)
                        try_times -= 1

                        new_mutation = None
                        if random_int == 1:
                            if sensitive:
                                new_mutation = fcg.build_mutation_add_edge_sensitive()
                            else:
                                new_mutation = fcg.build_mutation_add_edge()

                        

                        if new_mutation is not None:
                            state = fcg.process_mutation(new_mutation)
                            if state:
                                # add the new mutation
                                self.final_group_list.append([new_mutation])
                                break


    def update_dependency(self):
        flattened_list = [element for sublist in self.final_group_list for element in sublist]
        self.mutations = flattened_list
        #1.clear the last data
        self.dict_mutationIndex_to_group = {}
        self.dict_addNode_to_mutationIndex = {}
        self.dict_addEdge_to_mutationIndex = {}
        self.dict_removeEdge_to_mutationIndex = {}
        self.group_list = []
        #2.update the data, the resutl is final_group_list
        self.generate_group_list()

    def save_mutations(self):
        # 保存为json文件
        json_string = json.dumps(self.mutations)

        # 将二进制数据写入文件
        file_name = str(time.time()) + '_mutations.json'
        with open(file_name, 'a') as file:
            file.write(json_string + '\n')



    def save_group_list(self, id):
        #保存为json文件
        json_string = json.dumps(self.group_list)

        # 将二进制数据写入文件
        file_name = str(id) + '.json'
        with open(file_name, 'a') as file:
            file.write(json_string + '\n')
        # print("save mutation log successfully!")

    def read_group_list(self, id):
        file_name = str(id) + '.json'
        if not os.path.exists(file_name):
            return None

        loaded_data = []

        # 从文件中逐行读取JSON字符串并反序列化为字典
        with open(file_name, 'r') as file:
            for line in file:
                data = json.loads(line.strip())  # 移除行尾的换行符并反序列化
                loaded_data.append(data)
        # print(loaded_data)
        self.group_list = loaded_data[0]
        return loaded_data[0]

    def clear_group_list(self,id):
        file_name = str(id) + '.json'

        # 使用写入模式打开文件并覆盖内容
        with open(file_name, 'w') as file:
            file.write('')







if __name__ == '__main__':
    # fcg = FCG()
    # fcg.load(
    #     '/data/c/shiwensong/Malscan/MalScan-code/benign_2018_gexf/FFE2031B63A7ED452226D44F87F47B43288C3573DC0EAB4B0C7C7F2597DDC16B.gexf',
    #     0)
    # # print(fcg.edges)
    #
    # mutation = Mutation(fcg)
    # log = mutation.read_log() #500次测试
    
    # if ('Lcom/unity3d/plugin/downloader/b/a;->a(Lorg/apache/http/client/methods/HttpUriRequest;)Ljava/lang/String', 'Ljava/io/ByteArrayOutputStream;->toString()Ljava/lang/String;') in fcg.edges:
    #     print("yes")
    # print("mutation ing edges", len(mutation.edges), len(fcg.edges))
    # print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))
    # mutation.clear_log()
    # for i in range(500):
    #     #生成随机数1-4
    #     print('iteration', i)
    #     print("mutation ing edges", len(mutation.edges), len(fcg.edges))
    #     print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))
    #     random_int = random.randint(1, 4)
    #     print(random_int)
    #     if random_int == 1:
    #         mutation.add_edge()
    #     elif random_int == 2:
    #         mutation.add_node()
    #     elif random_int == 3:
    #         mutation.remove_node()
    #     elif random_int == 4:
    #         mutation.rewiring()
    # print("mutation ing edges", len(mutation.edges), len(fcg.edges))
    # print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))
    # print(mutation.mutation)

    # ('Lcom/unity3d/plugin/downloader/c/j;->a(Ljava/lang/String;)Ljava/security/PublicKey;', 'Ljava/security/KeyFactory;->getInstance(Ljava/lang/String;)Ljava/security/KeyFactory;')

    # fcg.process_mutation(mutation.mutation)
    # fcg.save('/data/c/shiwensong/Malscan/MalScan-code/test_coding')


    #对序列进行操作
    log = [
        {'feature_type': 'add_edge',
         'add_edges': [['1', '2']],
         'remove_edges': [],
         'add_nodes': [],
         'remove_nodes': []
         },
        {'feature_type': 'add_edge',
         'add_edges': [['5', '6']],
         'remove_edges': [],
         'add_nodes': [],
         'remove_nodes': []
         },
        {'feature_type': 'rewiring',
         'add_edges': [['5', '7'], ['7', '6']],
         'remove_edges': [['5', '6']],
         'add_nodes': [],
         'remove_nodes': []
         },
        {'feature_type': 'add_edge',
         'add_edges': [['5', '6']],
         'remove_edges': [],
         'add_nodes': [],
         'remove_nodes': []
         },
        {'feature_type': 'rewiring',
         'add_edges': [['5', '8'], ['8', '6']],
         'remove_edges': [['5', '6']],
         'add_nodes': [],
         'remove_nodes': []
         },
        {'feature_type': 'add_node',
         'add_edges': [['2', '3']],
         'remove_edges': [], 'add_nodes': ['3'], 'remove_nodes': []},
        {'feature_type': 'rewiring',
         'add_edges': [['2', '4'],
                       ['4', '3']],
         'remove_edges': [['2', '3']],
         'add_nodes': [], 'remove_nodes': []},
        {'feature_type': 'remove_node',
         'add_edges': [['1', '4']],
         'remove_edges': [['1', '2'],
                          ['2', '4']],
         'add_nodes': [],
         'remove_nodes': ['2']}
    ]

    mutationSequence = MutationSequence(log)
    mutationSequence.get_mutation_to_group()
    mutationSequence.print_group_list()
    # mutationSequence.merge_parent_group([0, 1])
    #

    # print("log_len", len(log))
    # log = mutation.read_log()
    # mutationSequence.generate_group_list(log)

    # print("group_list", len(mutationSequence.group_list))
    # mutationSequence.clear_group_list(1)
    # mutationSequence.save_group_list(1)
    # res = mutationSequence.read_group_list(1)
    # print("len group list", len(res))
    # print(res)

    # #random操作group_list
    # demo_group_list = mutationSequence.get_effect_group_list()
    # #随机打乱
    # random.shuffle(demo_group_list)
    # for i in range(len(demo_group_list)):
    #     print("group", i)
    #     print("len", len(demo_group_list[i]))
    #     print("dependency")
    #     for j in range(len(demo_group_list[i])):
    #         print(demo_group_list[i][j])
    #     print("------------------")

    #测试demo时使用，random交换是否可以修改FCG
    # demo_fcg = FCG()
    # demo_fcg.load(
    #     '/data/c/shiwensong/Malscan/MalScan-code/benign_2018_gexf/FFE2031B63A7ED452226D44F87F47B43288C3573DC0EAB4B0C7C7F2597DDC16B.gexf',
    #     0)
    # demo_fcg.init_call_graph()
    # print("demo_fcg", len(demo_fcg.edges), len(demo_fcg.nodes))
    # demo_fcg.cal_centralities()
    # degree = demo_fcg.degree_feature
    # katz = demo_fcg.katz_feature
    # closeness = demo_fcg.closeness_feature
    # harmonic = demo_fcg.harmonic_feature
    # # demo_fcg.init_centralities()
    # for i in range(len(demo_group_list)):
    #     print("group", i)
    #     demo_fcg.process_mutation_sequence(demo_group_list[i])
    #     print("demo_fcg_after_mutation", len(demo_fcg.edges), len(demo_fcg.nodes))
    # demo_fcg.cal_centralities()
    # degree_after = demo_fcg.degree_feature
    # katz_after = demo_fcg.katz_feature
    # closeness_after = demo_fcg.closeness_feature
    # harmonic_after = demo_fcg.harmonic_feature
    #
    # cnt_degree = 0
    # cnt_katz = 0
    # cnt_closeness = 0
    # cnt_harmonic = 0
    # #比较前后特征
    # for i in range(len(degree)):
    #     if degree[i] != degree_after[i]:
    #         # print("degree", i)
    #         cnt_degree += 1
    #
    # for i in range(len(katz)):
    #     if katz[i] != katz_after[i]:
    #         # print("katz", i)
    #         cnt_katz += 1
    #
    # for i in range(len(closeness)):
    #     if closeness[i] != closeness_after[i]:
    #         # print("closeness", i)
    #         cnt_closeness += 1
    #
    # for i in range(len(harmonic)):
    #     if harmonic[i] != harmonic_after[i]:
    #         # print("harmonic", i)
    #         cnt_harmonic += 1
    #
    # print("cnt_degree", cnt_degree)
    # print("cnt_katz", cnt_katz)
    # print("cnt_closeness", cnt_closeness)
    # print("cnt_harmonic", cnt_harmonic)

    #测试ga_mutation
    # mutationSequence.ga_mutation()

    #166 167


