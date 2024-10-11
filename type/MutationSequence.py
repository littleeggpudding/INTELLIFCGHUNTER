import time
import random
import json
import os
import copy

class MutationSequence:

    def __init__(self, mutations, state=False):
        self.mutations = mutations
        self.group_list = []  # Stores the index of mutations, each group is a list
        self.final_group_list = []  # Stores mutation objects
        self.dict_addNode_to_mutationIndex = {}  # Tracks the relationship between newly added nodes and mutation indices
        self.dict_addEdge_to_mutationIndex = {}  # Tracks the relationship between newly added edges and mutation indices
        self.dict_removeEdge_to_mutationIndex = {}  # Tracks the relationship between removed edges and mutation indices
        self.dict_mutationIndex_to_group = {}  # Tracks the relationship between mutations and groups

        if not state:
            # If state == False, generate groups with dependency between mutations
            self.generate_group_list()
        else:
            # Ignore dependencies between mutations
            self.final_group_list = [[mutation] for mutation in mutations]

    def generate_group_list(self):


        for i in range(len(self.mutations)):
            mutation = self.mutations[i] 
            mutation = mutation.mutation
            current_type = mutation.get('feature_type', '')

            if current_type == 'add_node':
                parent_index = self._find_parent_add_node(mutation)

                if parent_index != -1:
                    # Find the group corresponding to the parent and add the current mutation to the group
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
                # There may be multiple parents
                parents_indexes = self._find_parent_add_edge(mutation)

                if len(parents_indexes) != 0:
                    if len(parents_indexes) == 1:
                        # Find the group corresponding to the parent and add the current mutation to the group
                        group = self.dict_mutationIndex_to_group[parents_indexes[0]]
                        group.append(i)
                        self.dict_mutationIndex_to_group[i] = group
                    else:
                        # Has multiple parent nodes and needs to be merged
                        # All merged into the group of the first parent node
                        groups = self.merge_parent_group(parents_indexes)
                        merged_group = groups[0]

                        for group in groups[1:]:
                            for mutation_id in group:
                                # Each mutation is added to the group of the first parent node and updates the dict_node_to_group
                                merged_group.append(mutation_id)
                                self.dict_mutationIndex_to_group[mutation_id] = merged_group
                            # Delete group from group_list
                            self.group_list.remove(group)

                        # For the current mutation, it is also added to the group of the first parent node
                        merged_group.append(i)
                        self.dict_mutationIndex_to_group[i] = merged_group
                else:
                    group = [i]
                    self.group_list.append(group)
                    self.dict_mutationIndex_to_group[i] = group

                add_edge = mutation.get('add_edges', [])[0]
                add_edge = json.dumps(add_edge)
                self.dict_addEdge_to_mutationIndex[add_edge] = i

            elif current_type == 'rewiring':
                parents_indexes = self._find_parent_rewiring(mutation)

                if len(parents_indexes) != 0:
                    if len(parents_indexes) == 1:
                        group = self.dict_mutationIndex_to_group[parents_indexes[0]]
                        group.append(i)
                        self.dict_mutationIndex_to_group[i] = group
                    else:

                        groups = self.merge_parent_group(parents_indexes)
                        merged_group = groups[0]

                        for group in groups[1:]:
                            for mutation_id in group:

                                merged_group.append(mutation_id)
                                self.dict_mutationIndex_to_group[mutation_id] = merged_group

                            self.group_list.remove(group)

                        merged_group.append(i)
                        self.dict_mutationIndex_to_group[i] = merged_group
                else:
                    group = [i]
                    self.group_list.append(group)
                    self.dict_mutationIndex_to_group[i] = group

                remove_edge = mutation.get('remove_edges', [])[0]
                remove_edge = json.dumps(remove_edge)
                self.dict_removeEdge_to_mutationIndex[remove_edge] = i

                add_edges = mutation.get('add_edges', []) # Not sure how many edges there are
                for add_edge in add_edges:
                    add_edge = json.dumps(add_edge)
                    self.dict_addEdge_to_mutationIndex[add_edge] = i

            elif current_type == 'remove_node':

                parents_indexes = self._find_parent_remove_node(mutation)


                if len(parents_indexes) != 0:
                    if len(parents_indexes) == 1:

                        group = self.dict_mutationIndex_to_group[parents_indexes[0]]
                        group.append(i)
                        self.dict_mutationIndex_to_group[i] = group
                    else:

                        groups = self.merge_parent_group(parents_indexes)
                        merged_group = groups[0]

                        for group in groups[1:]:

                            for mutation_id in group:

                                merged_group.append(mutation_id)
                                self.dict_mutationIndex_to_group[mutation_id] = merged_group

                            self.group_list.remove(group)

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

        return groups


    def print_group_list(self):
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


    def _find_parent_add_node(self, current):

        #check:
        #1.begin node is a new added node
        add_edge = current.get('add_edges', [])[0]
        begin_node = add_edge[0]
        #If there is no key==begin node, -1 is returned
        begin_node = json.dumps(begin_node)
        index = self.dict_addNode_to_mutationIndex.get(begin_node, -1)
        return index

    def _find_parent_add_edge(self, current):
        # check:
        # 1. begin node is a newly added point
        # 2. end node is a newly added point
        # 3. edge is an edge with a removed before
        parents_indexes = set()

        add_edge = current.get('add_edges', [])[0]

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

        parents_indexes = set()

        remove_edge = current.get('remove_edges', [])[0]
        add_edges = current.get('add_edges', [])
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

        # check:
        # remove edge is a new added edge
        remove_edge = json.dumps(remove_edge)
        index = self.dict_addEdge_to_mutationIndex.get(remove_edge, -1)
        if index != -1:
            parents_indexes.add(index)

        if mid_node != '':
            mid_node = json.dumps(mid_node)
            index = self.dict_addNode_to_mutationIndex.get(mid_node, -1)
            if index != -1:
                parents_indexes.add(index)

        return list(parents_indexes)

    def _find_parent_remove_node(self, current):
        mid_node = current.get('remove_nodes', [])[0]

        parents_indexes = set()

        # 3.mid node is a new added node
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

        cnt_1 = 0
        cnt_0 = 0
        for i in range(min_length):
            choose = random.randint(0, 1)
            if choose == 0:

                new_mutation_list.append(mutation_list1[i])
                cnt_0 += 1

            else:

                new_mutation_list.append(mutation_list2[i])
                cnt_1 += 1


        self.final_group_list = new_mutation_list

    def solve_conflict(self, fcg, mutation, res):
        #1.mutation
        type = mutation.mutation.get('feature_type', '')
        res_type = next(iter(res))


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
        # The mutation corresponding to index
        for i in range(len(group_list)):
            group_mutation = []
            # deal with a group
            for j in range(len(group_list[i])):
                index = group_list[i][j]
                group_mutation.append(self.mutations[index])
            mutation_list.append(group_mutation)

        return mutation_list

    def _concatenate(self, mutation_list1, mutation_list2):

        new_mutation_list = []
        new_mutation_list.extend(mutation_list1)
        new_mutation_list.extend(mutation_list2)
        return new_mutation_list

    def ga_mutation(self, fcg, mutation_pro = 0.2, sensitive = True):

        group_list_length = len(self.final_group_list)


        for i in range(group_list_length):
            group = self.final_group_list[i]
            group_length = len(group)
            if group_length == 0:
                continue


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


    def ga_mutation_add_edge(self, fcg, mutation_pro = 0.2, sensitive = True):
        #fcg: current fcg

        print("ga_mutation ing")
        group_list_length = len(self.final_group_list)

        for i in range(group_list_length):
            group = self.final_group_list[i]
            group_length = len(group)
            if group_length == 0:
                continue


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
        json_string = json.dumps(self.mutations)
        file_name = str(time.time()) + '_mutations.json'
        with open(file_name, 'a') as file:
            file.write(json_string + '\n')



    def save_group_list(self, id):
        json_string = json.dumps(self.group_list)
        file_name = str(id) + '.json'
        with open(file_name, 'a') as file:
            file.write(json_string + '\n')

    def read_group_list(self, id):
        file_name = str(id) + '.json'
        if not os.path.exists(file_name):
            return None

        loaded_data = []


        with open(file_name, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                loaded_data.append(data)

        self.group_list = loaded_data[0]
        return loaded_data[0]

    def clear_group_list(self,id):
        file_name = str(id) + '.json'


        with open(file_name, 'w') as file:
            file.write('')


