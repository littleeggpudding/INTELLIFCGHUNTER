import tensorflow as tf
import joblib
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json
import sys
import os
import copy
from torch_geometric.data import Data
import random
import numpy as np
sys.path.append(os.path.abspath('../'))
from Other.Classify import GCN
from type.MutationSequence import MutationSequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    # Load the model from the specified path
    if 'MLP' in model_path:
        MLP_model = tf.keras.models.load_model(model_path)
        return MLP_model
    else:
        model = joblib.load(model_path)
        return model
    
def load_GCN_model(path):
    # Initialize the model with the same architecture as used during training
    gcn_model = GCN(num_node_features=2, num_classes=2)

    # Load the model weights into CPU first, then transfer to device if necessary

    model_weights = torch.load(path, map_location=torch.device('cpu'))
    gcn_model.load_state_dict(model_weights)

    # Set the model to evaluation mode before using it for predictions
    gcn_model.eval()

    # Transfer the model to the appropriate device
    gcn_model.to(device)
    return gcn_model

def load_MLP_model(model_path):
    # Define a new model structure, identical to the original model,
    # but the last Dense layer has no activation function
    MLP2 = Sequential([
        Dense(128, activation='relu', input_shape=(2704,)),  # Same input shape
        Dense(64, activation='relu'),
        Dense(1)  # No activation function in the final layer
    ])

    # Load the original model
    original_model = tf.keras.models.load_model(model_path)

    for layer, original_layer in zip(MLP2.layers, original_model.layers):
        layer.set_weights(original_layer.get_weights())

    return MLP2


def save_log_for_every_individual(fcg, individual, score, generation, save_dir):
    extracted_mutations = []
    for group in individual.final_group_list:
        group_list = []
        for mutation in group:
            group_list.append(mutation.mutation)
        extracted_mutations.append(group_list)

    data = {
        'individual': extracted_mutations,
        'score': str(score),
        'generation': generation
    }

    try:
        json_string = json.dumps(data)
        file_name = f'{save_dir}/{fcg.apk_name}_log_for_every_individual.json'
        with open(file_name, 'a') as file:
            file.write(json_string + '\n')
        # print("save mutation log successfully!")
    except Exception as e:
        print("Failed to save log:", str(e))


def save_graph_for_every_generation(fcg, state, generation, save_dir):
    if state == 'failed':
        fcg.save(f'{save_dir}/ga_failed/', generation)
    elif state == 'success':
        fcg.save(f'{save_dir}/ga_success/', generation)


def save_log_for_every_generation(fcg, score, shap, generation, population, save_dir):
    individual_list = []
    for individual in population:
        extracted_mutations = []
        for group in individual.final_group_list:
            group_list = []
            for mutation in group:
                group_list.append(mutation.mutation)
            extracted_mutations.append(group_list)
        individual_list.append(extracted_mutations)

    data = {
        'score': str(score),
        'shap': str(shap),
        'generation': generation,
        'individual_list': individual_list
    }
    try:
        json_string = json.dumps(data)
        print("save log for every generation")
        file_name = f'{save_dir}/{fcg.apk_name}_log_for_every_generation.json'
        with open(file_name, 'a') as file:
            file.write(json_string + '\n')
        # print("save mutation log successfully!")
    except Exception as e:
        print("Failed to save log:", str(e))


def print_fcg_information(fcg):
    print("fcg nodes", len(fcg.nodes))
    print("fcg edges", len(fcg.edges))
    print("fcg boundary edges", len(fcg.boundary_edges))
    print("fcg system nodes", len(fcg.system_nodes))
    print("fcg apk name", fcg.apk_name)


def test_GCN_model(data, sub_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = sub_model.to(device)

    with torch.no_grad():
        data = data.to(device)
        out = model(data)
        score = out[0][0].item() - out[0][1].item()  # benign - malware, the larger the better

    return score


def remap_indices(edges, nodes):
    # Create a new mapping for node indices
    unique_nodes = sorted(set(nodes))
    mapping = {node: i for i, node in enumerate(unique_nodes)}

    # Update edge indices to reflect new continuous node indices
    remapped_edges = [[mapping[edge[0]], mapping[edge[1]]] for edge in edges if
                      edge[0] in mapping and edge[1] in mapping]

    return remapped_edges, unique_nodes, mapping


def obtain_gcn_feature(fcg):
    fcg.generate_new_call_graph()

    # Collect initial edges and nodes from the graph
    original_edges = [[edge[0], edge[1]] for edge in fcg.edges]
    nodes = list(fcg.nodes)

    # Obtain union of nodes involved in edges to ensure all are accounted for
    edge_nodes = set(sum(original_edges, []))  # Flatten list of edges and create a set of nodes
    all_nodes = sorted(set(nodes).union(edge_nodes))  # Union of explicit nodes and edge nodes

    # Remap node indices for continuity and update edge indices
    remapped_edges, remapped_nodes, mapping = remap_indices(original_edges, all_nodes)

    # Prepare edge index and node features
    edge_index = torch.tensor(remapped_edges, dtype=torch.long).contiguous()
    degree_features = []
    for node in remapped_nodes:
        if node in fcg.current_call_graph:
            in_degree = fcg.current_call_graph.in_degree(node)
            out_degree = fcg.current_call_graph.out_degree(node)
            degree_features.append([in_degree, out_degree])
        else:
            degree_features.append([0, 0])  # Assign zero degrees if node is missing in the graph

    degree_features = torch.tensor(degree_features, dtype=torch.float)
    data = Data(x=degree_features, edge_index=edge_index)

    return data


# for initial population
def random_m(fcg, steps):
    """
        Applies a series of random mutations to a function call graph (FCG) and returns the modified graph.

        Parameters:
        fcg : object
            The original function call graph (FCG) to be mutated.
        steps : int
            The number of mutation steps to be applied.

        Returns:
        mutation_list : list
            A list of mutations applied to the FCG.
        fcg : object
            The mutated function call graph after applying the mutations.
    """
    fcg = copy.deepcopy(fcg)
    mutation_list = []
    visited_random_number = set()

    cnt_random_1 = 0
    cnt_random_2 = 0
    cnt_random_3 = 0
    cnt_random_4 = 0

    # To avoid the user nodes are not enough to opearte the mutation
    for i in range(100):
        mutation = fcg.build_mutation_add_node()
        state, res = fcg.process_mutation(mutation)
        if state:
            mutation_list.append(mutation)

    i = 0
    while i < steps:
        random_int = 2 # only add edge in BagAmoo


        mutation = None
        if random_int == 1:
            mutation = fcg.build_mutation_add_node()
            cnt_random_1 += 1
        elif random_int == 2:
            mutation = fcg.build_mutation_add_edge()
            cnt_random_2 += 1
        elif random_int == 3:
            mutation = fcg.build_mutation_rewiring()
            cnt_random_3 += 1
        elif random_int == 4:
            mutation = fcg.build_mutation_remove_node()
            cnt_random_4 += 1

        if mutation is None:
            if random_int == 1:
                cnt_random_1 -= 1
            elif random_int == 2:
                cnt_random_2 -= 1
            elif random_int == 3:
                cnt_random_3 -= 1
            elif random_int == 4:
                cnt_random_4 -= 1

            visited_random_number.add(random_int)
            if visited_random_number == {1, 2, 3, 4}:
                print("all mutations failed")
                break

        else:
            state, res = fcg.process_mutation(mutation)
            if state:
                mutation_list.append(mutation)
            else:
                continue

        i += 1

    return mutation_list, fcg

def random_m_our(fcg, steps, sensitive):
    fcg = copy.deepcopy(fcg)

    mutation_list = []
    visited_random_number = set()

    cnt_random_1 = 0 #add node
    cnt_random_2 = 0  # add edge 
    cnt_random_3 = 0 #rewire
    cnt_random_4 = 0 #remove node

    for i in range(100):
        mutation = fcg.build_mutation_add_node()
        state, res = fcg.process_mutation(mutation)
        if state:
            mutation_list.append(mutation)

    i = 0
    while i < steps:

        random_int = random.randint(1, 4)

        mutation = None
        if random_int == 1:
            mutation = fcg.build_mutation_add_node()
            cnt_random_1 += 1
        elif random_int == 2:
            if sensitive:
                mutation = fcg.build_mutation_add_edge_sensitive()
            else:
                mutation = fcg.build_mutation_add_edge()
            cnt_random_2 += 1
        elif random_int == 3:
            if sensitive:
                mutation = fcg.build_mutation_rewiring_sensitive()
            else:
                mutation = fcg.build_mutation_rewiring()
            cnt_random_3 += 1
        elif random_int == 4:
            if sensitive:
                mutation = fcg.build_mutation_remove_node_sensitive()
            else:
                mutation = fcg.build_mutation_remove_node()
            cnt_random_4 += 1

        if mutation is None:
            if random_int == 1:
                cnt_random_1 -= 1
            elif random_int == 2:
                cnt_random_2 -= 1
            elif random_int == 3:
                cnt_random_3 -= 1
            elif random_int == 4:
                cnt_random_4 -= 1
          

            visited_random_number.add(random_int)
            if visited_random_number == {1, 2, 3, 4}:
                print("all mutations failed")
                break

        else:
            state, res = fcg.process_mutation(mutation)
            if state:
                mutation_list.append(mutation)
            else:
                continue

        i += 1
    return mutation_list, fcg


def _init_population(fcg, pop_num=100, steps=500):
    # Randomly generate a pop num mutation sequence
    population = []
    try_times = 0
    while try_times < pop_num:
        mutation_lists, cur_fcg = random_m(fcg, steps)

        mutationSequence = MutationSequence(mutation_lists)
        population.append(mutationSequence)
        try_times += 1

    return population

def _init_population_our(fcg, pop_num=100, steps = 500, sensitive = True, group = False):
    
    # sensitive=True: use sensitive mutation
    # group=False: use dependency-aware mutations

    population = []
    try_times = 0
    while try_times < pop_num:
        mutation_lists, cur_fcg = random_m_our(fcg, steps, sensitive)

        mutationSequence = MutationSequence(mutation_lists, group)
        population.append(mutationSequence)
        try_times += 1

    return population


def deal_with_conflict(fcg, individual1, control_group = False):
    new_fcg = copy.deepcopy(fcg)

    ori_conflict = 0
    now_conflict = 0
    null_group_idx = []
    for j in range(len(individual1.final_group_list)):
        group = individual1.final_group_list[j]
        need_remove = []
        for i in range(len(group)):
            mutation = group[i]
            state, res = new_fcg.process_mutation(mutation)
            if not state:
                if not control_group:
                    state_conflict = individual1.solve_conflict(new_fcg, mutation, res)
                    if not state_conflict:
                        # discard mutation
                        need_remove.append(i)
                        now_conflict += 1
                    ori_conflict += 1
                else:
                    need_remove.append(i)
                    ori_conflict += 1

        for idx in need_remove[::-1]:
            del group[idx]

        if len(group) == 0:
            null_group_idx.append(j)

    for idx in null_group_idx[::-1]:
        individual1.final_group_list.pop(idx)


    return new_fcg



# To caluate average feature
def cal_4features_avg(degree_vector, katz_vector, closeness_vector, harmonic_vector):
    avg_vector = []
    for i in range(len(degree_vector)):
        avg_feature = (degree_vector[i] + katz_vector[i] + closeness_vector[i] + harmonic_vector[i]) / 4.0
        avg_vector.append(avg_feature)

    return np.array(avg_vector)


# To caluate the opposite adjustment using shape value
def calculate_opposite_adjustment(original_feature, now_feature, shap_value):
    negative_shap_value = []
    negative_diff_feature = []
    positive_shap_value = []
    positive_diff_feature = []
    for i in range(len(shap_value)):
        if shap_value[i] < 0:
            negative_shap_value.append(shap_value[i])
            negative_diff_feature.append(now_feature[i] - original_feature[i])
        else:
            positive_shap_value.append(shap_value[i])
            positive_diff_feature.append(now_feature[i] - original_feature[i])

    neg_index = np.argsort(negative_shap_value)[::-1]
    pos_index = np.argsort(positive_shap_value)

    opposite_adjustment = 0.0
    for index in neg_index:
        opposite_adjustment = opposite_adjustment - negative_shap_value[index] * negative_diff_feature[index]

    for index in pos_index:
        opposite_adjustment = opposite_adjustment - positive_shap_value[index] * positive_diff_feature[index]


    return opposite_adjustment


# For attack tree-based model
def process_tree_based_model(fcg, attack_model, feature_vector, feature_type):
    # Get the indices of non-zero features
    non_zero_indices = [i for i in range(len(feature_vector[0])) if feature_vector[0][i] != 0]

    # Set to store important features from decision trees
    important_features = set()

    # Iterate over each tree in the ensemble model
    for estimator in attack_model.estimators_:
        tree = estimator.tree_
        features = tree.feature
        node_queue = [(0, [])]  # Queue to store (node_index, key_features)

        while node_queue:
            node_index, key_features = node_queue.pop(0)

            if features[node_index] == -2:  # Leaf node
                # If benign samples dominate this leaf, collect the features leading to it
                if tree.value[node_index][0][0] - tree.value[node_index][0][1] > 0:
                    important_features.update(key_features)
                continue

            if features[node_index] in non_zero_indices:
                # Traverse left and right branches if the feature is non-zero
                left_index, right_index = tree.children_left[node_index], tree.children_right[node_index]
                split_value = tree.threshold[node_index]
                current_value = feature_vector[0][features[node_index]]

                # Append conditions to key_features
                left_conditions = key_features + [(features[node_index], split_value, 1)]
                right_conditions = key_features + [(features[node_index], split_value, -1)]

                node_queue.append((left_index, left_conditions))
                node_queue.append((right_index, right_conditions))
            else:
                # Skip non-relevant features
                if feature_vector[0][features[node_index]] < tree.threshold[node_index]:
                    node_queue.append((tree.children_left[node_index], key_features))
                else:
                    node_queue.append((tree.children_right[node_index], key_features))

    # Initialize bounds for each important feature
    upper_bounds = {}
    lower_bounds = {}

    # Set default bounds based on feature type
    for feature in important_features:
        idx, threshold, sign = feature

        if idx < 3 * 430 and feature_type != "harmonic":
            upper_bounds[idx], lower_bounds[idx] = [1], [0]
        else:
            upper_bounds[idx], lower_bounds[idx] = [4000], [0]

        # Adjust bounds based on the condition (sign)
        if sign == 1:
            upper_bounds[idx].append(threshold)
        else:
            lower_bounds[idx].append(threshold)

    # Refine bounds by removing redundant bounds
    refine_bounds(upper_bounds, lower_bounds)

    # Create attack vectors by trying different feature combinations within bounds
    attackable, not_sure, attack_bounds = generate_attack_vectors(upper_bounds, lower_bounds, feature_vector, attack_model)

    return attackable, not_sure, attack_bounds


def refine_bounds(upper_bounds, lower_bounds):
    """Refines upper and lower bounds by eliminating redundant values."""
    for idx in upper_bounds.keys():
        ori_upper_bounds = upper_bounds[idx]
        ori_lower_bounds = lower_bounds[idx]
        max_lower_bound = max(ori_lower_bounds)
        min_upper_bound = min(ori_upper_bounds)

        reserved_upper_bound = max(ori_upper_bounds)
        new_upper_bounds = [ub for ub in ori_upper_bounds if ub <= max_lower_bound]
        new_upper_bounds.append(min(ub for ub in ori_upper_bounds if ub > max_lower_bound))

        reserved_lower_bound = min(ori_lower_bounds)
        new_lower_bounds = [lb for lb in ori_lower_bounds if lb >= min_upper_bound]
        new_lower_bounds.append(max(lb for lb in ori_lower_bounds if lb < min_upper_bound))

        upper_bounds[idx], lower_bounds[idx] = new_upper_bounds, new_lower_bounds


def generate_attack_vectors(upper_bounds, lower_bounds, feature_vector, attack_model):
    """Generates potential attack feature vectors and determines attackability."""
    max_attempts = 20000
    epsilon = 1e-8
    attack_feature = feature_vector.copy()
    attack_bounds = [(-1, -1)] * len(attack_feature[0])
    conflict_values = []
    total_combinations = 1
    not_sure = False

    for idx in upper_bounds.keys():
        u_bounds = sorted(upper_bounds[idx])
        l_bounds = sorted(lower_bounds[idx])
        combined_bounds = sorted(u_bounds + l_bounds)

        values_to_try, value_bounds = [], []

        for i in range(len(combined_bounds) - 1):
            if combined_bounds[i] != combined_bounds[i + 1]:
                midpoint = (combined_bounds[i] + combined_bounds[i + 1]) / 2
                values_to_try.append(midpoint)
                value_bounds.append((combined_bounds[i], combined_bounds[i + 1]))

        if total_combinations * len(values_to_try) < max_attempts:
            total_combinations *= len(values_to_try)
            conflict_values.append((idx, values_to_try, len(values_to_try), value_bounds))
        else:
            not_sure = True

    attackable = False
    target_feature, target_bounds = None, None
    num_trials = min(max_attempts, total_combinations)
    attack_features = [attack_feature.copy() for _ in range(num_trials)]
    attack_bounds_list = [attack_bounds.copy() for _ in range(num_trials)]

    # Iterate through the possible feature combinations
    for i in range(num_trials):
        current_feature = attack_features[i]
        current_bounds = attack_bounds_list[i]
        score = attack_model.predict(current_feature)
        if len(score.shape) > 1:
            score = score.reshape(-1)
        if score[0] < 0.5:
            attackable = True
            target_feature = current_feature
            target_bounds = current_bounds
            break

    return attackable, not_sure, target_bounds


def process_tree_based_model_mamadroid_apigraph(fcg, att_model, ori_feature):
    """
    Process and extract the mutable features from the fcg for the tree-based model (e.g., Adaboost or RandomForest).
    Evaluate the feasibility of an attack by calculating bounds for critical features.
    """

    # Step 1: Identify mutable feature indexes (self-defined and obfuscated types)
    mutable_indexes = []
    types_num = len(fcg.type_count)

    # Process self-defined type
    if fcg.type_count[types_num - 2] > 0:
        for i in range(types_num):
            if fcg.type_count[i] > 0:
                # Non-zero edge from self-defined to type 'i', hence mutable
                index = (types_num - 2) * types_num + i
                mutable_indexes.append(index)

    # Process obfuscated type
    if fcg.type_count[types_num - 1] > 0:
        for i in range(types_num):
            if fcg.type_count[i] > 0:
                # Non-zero edge from obfuscated to type 'i', hence mutable
                index = (types_num - 1) * types_num + i
                mutable_indexes.append(index)

    # Step 2: Identify critical features using the tree structure of the model
    tree_key_features = set()

    for estimator in att_model.estimators_:
        tree_ = estimator.tree_
        features = tree_.feature
        queue = [(0, [])]  # (index, key_features)

        while queue:
            index, key_features = queue.pop(0)

            if features[index] == -2:  # Leaf node
                if tree_.value[index][0][0] > tree_.value[index][0][1]:  # More benign samples
                    tree_key_features.update(key_features)
                continue

            if features[index] in mutable_indexes:
                # Process mutable feature
                left_index = tree_.children_left[index]
                right_index = tree_.children_right[index]
                split_value = tree_.threshold[index]

                left_condition = key_features + [(features[index], split_value, 1)]
                right_condition = key_features + [(features[index], split_value, -1)]

                queue.append((left_index, left_condition))
                queue.append((right_index, right_condition))
            else:
                # Process immutable feature
                next_index = tree_.children_left[index] if ori_feature[0][features[index]] < tree_.threshold[index] else \
                tree_.children_right[index]
                queue.append((next_index, key_features))

    # Step 3: Adjust bounds based on the critical features
    upper_bounds_dict = {}
    lower_bounds_dict = {}

    for feature in tree_key_features:
        index, value, sign = feature
        if index not in upper_bounds_dict:
            upper_bounds_dict[index] = [1]
            lower_bounds_dict[index] = [0]

        if sign == 1:
            upper_bounds_dict[index].append(value)
        else:
            lower_bounds_dict[index].append(value)

    # Step 4: Refine bounds by discarding unnecessary extreme values
    for index in upper_bounds_dict.keys():
        ori_upper_bounds = upper_bounds_dict[index]
        ori_lower_bounds = lower_bounds_dict[index]

        max_lower_bound = max(ori_lower_bounds)
        min_upper_bound = min(ori_upper_bounds)

        reserved_upper_bound = min(upper_bound for upper_bound in ori_upper_bounds if upper_bound > max_lower_bound)
        reserved_lower_bound = max(lower_bound for lower_bound in ori_lower_bounds if lower_bound < min_upper_bound)

        upper_bounds_dict[index] = [bound for bound in ori_upper_bounds if bound <= max_lower_bound] + [
            reserved_upper_bound]
        lower_bounds_dict[index] = [bound for bound in ori_lower_bounds if bound >= min_upper_bound] + [
            reserved_lower_bound]

    # Step 5: Attempt to find attackable features within the calculated bounds
    attack_feature = copy.deepcopy(ori_feature)
    attack_bounds = [(-1, -1)] * len(attack_feature[0])
    conflict_values = []
    total_trys = 1
    max_try = 20000
    epsilon = 1e-8
    not_sure = False

    for index in upper_bounds_dict.keys():
        upper_bounds = upper_bounds_dict[index]
        lower_bounds = lower_bounds_dict[index]

        if len(upper_bounds) == 1 and len(lower_bounds) == 1:
            upper_bound = upper_bounds[0]
            lower_bound = lower_bounds[0]
            attack_feature[0][index] = lower_bound + epsilon if upper_bound == 4000 else (lower_bound + upper_bound) / 2
            attack_bounds[index] = (lower_bound, upper_bound)
        else:
            # Handle conflicting bounds
            bounds = sorted(upper_bounds + lower_bounds)
            values_to_try = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)]
            values_bounds = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

            if total_trys * len(values_to_try) < max_try:
                total_trys *= len(values_to_try)
                conflict_values.append((index, values_to_try, len(values_to_try), values_bounds))
            else:
                not_sure = True

    # Step 6: Fill in all possible attack features based on conflict values
    all_attack_features = [copy.deepcopy(attack_feature) for _ in range(total_trys)]
    all_attack_bounds = [copy.deepcopy(attack_bounds) for _ in range(total_trys)]
    circular_num = total_trys

    for conflict_value in conflict_values:
        circular_num //= conflict_value[2]
        values_to_try = conflict_value[1]
        values_bounds = conflict_value[3]
        fea_index = conflict_value[0]

        for out_i in range(int(total_trys // (circular_num * conflict_value[2]))):
            for mid_i in range(conflict_value[2]):
                for inner_i in range(int(circular_num)):
                    att_index = int(out_i * circular_num * conflict_value[2] + mid_i * circular_num + inner_i)
                    all_attack_features[att_index][0][fea_index] = values_to_try[mid_i]
                    all_attack_bounds[att_index][fea_index] = values_bounds[mid_i]

    # Step 7: Test attack feasibility
    attackable = False
    target_att_feature = None
    target_att_bounds = None
    trys = min(max_try, total_trys)

    for i in range(trys):
        att_feature = all_attack_features[i]
        att_score = att_model.predict(att_feature).reshape(-1)

        if att_score[0] < 0.5:
            attackable = True
            target_att_feature = att_feature
            target_att_bounds = all_attack_bounds[i]
            break

    return attackable, not_sure, target_att_bounds




