import sys
import os
import torch
import argparse
import random
import copy
import psutil
import glob


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from type.FCG_malscan_tree import FCG_malscan_tree as FCG
from type.Mutations import Mutations
from utils import load_model, process_tree_based_model
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)


#global variable
save_dir = None
target_model = None #target model
feature_type = None

pop_num = None
max_generation = None
steps = None

def ga(fcg, target_bounds, sparse_gene, dense_gene, longlink_times_gene,
                               longlink_len_gene, sensitive_nodes):
    # Initialize gene pools for mutation
    sparse_gene_pool = [int(sparse_gene * r) for r in [0.8, 0.9, 1.0, 1.1, 1.2]]
    dense_gene_pool = [int(dense_gene * r) for r in [0.8, 0.9, 1.0, 1.1, 1.2]]

    longlink_len_pool = {}
    longlink_times_pool = {}

    # Generate pools for longlink genes
    for i in range(len(longlink_times_gene)):
        if longlink_times_gene[i] == 0:
            continue
        longlink_len_pool[i] = random.choices(range(10), k=3) + [longlink_len_gene[i]]
        longlink_times_pool[i] = [int(longlink_times_gene[i] * r) for r in [0.5, 0.7, 1.0]]

    # Initialize population for the genetic algorithm
    population = []

    for _ in range(pop_num):
        sparse_gene = random.choice(sparse_gene_pool)
        dense_gene = random.choice(dense_gene_pool)
        longlink_times = [0] * len(sensitive_nodes)
        longlink_len = [0] * len(sensitive_nodes)

        for j in range(len(sensitive_nodes)):
            if j in longlink_times_pool:
                longlink_times[j] = random.choice(longlink_times_pool[j])
                longlink_len[j] = random.choice(longlink_len_pool[j])

        ms = Mutations(sparse_gene, dense_gene, longlink_times, longlink_len, sensitive_nodes)
        population.append(ms)


    # Begin GA process
    for gen in range(max_generation):
        # Calculate fitness for each individual in the population
        fitness = []
        for ms in population:
            feature = calc_feature(fcg, ms)
            att_score = target_model.predict(feature).reshape(-1)
            if att_score[0] < 0.5:
                # Attack successful, return the mutation and the generation number
                return True, ms, gen + 1

            # Calculate fitness score based on target bounds
            score = sum(1 for i, (lb, ub) in enumerate(target_bounds) if lb < feature[0][i] < ub)
            fitness.append(score)

        # Sort population by fitness in descending order
        population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0], reverse=True)]

        # Create next generation
        next_generation = []
        while len(next_generation) < pop_num:
            # Select two best individuals from random groups
            group_size = int(pop_num * 0.2)
            group1 = random.sample(range(pop_num), group_size)
            group2 = random.sample(range(pop_num), group_size)
            best1 = population[group1[0]]
            best2 = population[group2[0]]

            # Perform crossover
            sparse_gene = random.choice([best1.add_sparse_nodes_gene, best2.add_sparse_nodes_gene])
            dense_gene = random.choice([best1.add_density_nodes_gene, best2.add_density_nodes_gene])
            longlink_times = [random.choice([best1.longlink_times_gene[j], best2.longlink_times_gene[j]]) 
                              for j in range(len(sensitive_nodes))]
            longlink_len = [random.choice([best1.longlink_len_gene[j], best2.longlink_len_gene[j]]) 
                            for j in range(len(sensitive_nodes))]

            # Perform mutation with a 5% probability
            if random.random() < 0.05:
                sparse_gene = random.choice([int(sparse_gene * 0.9), int(sparse_gene * 1.1)])
                dense_gene = random.choice([int(dense_gene * 0.9), int(dense_gene * 1.1)])
                for j in range(len(sensitive_nodes)):
                    longlink_times[j] = random.choice([int(longlink_times[j] * 0.9), int(longlink_times[j] * 1.1)])
                    longlink_len[j] = random.choice([int(longlink_len[j] * 0.9), int(longlink_len[j] * 1.1)])

            ms = Mutations(sparse_gene, dense_gene, longlink_times, longlink_len, sensitive_nodes)
            next_generation.append(ms)

        # Replace population with the next generation
        population = next_generation

    # Return failure if no successful attack was found
    return False, None, -1

def mutation_with_guidance(fcg, target_att_bounds):
    if feature_type == 'degree':
        result,ms,gen = mutation_with_guidance_degree(fcg, target_att_bounds)
    elif feature_type == 'katz':
        result,ms,gen = mutation_with_guidance_katz(fcg, target_att_bounds)
    elif feature_type == 'closeness':
        result,ms,gen = mutation_with_guidance_closeness(fcg, target_att_bounds)
    elif feature_type == 'harmonic':
        result,ms,gen = mutation_with_guidance_harmonic(fcg, target_att_bounds)
    elif feature_type == 'concentrate':
        result,ms,gen = mutation_with_guidance_combine(fcg, target_att_bounds)
    elif feature_type == 'average':
        result,ms,gen = mutation_with_guidance_average(fcg, target_att_bounds)
    else:
        result,ms,gen = False,None,-1
    if result:
        save_mutated_fcg(fcg, ms, gen)
    return result, gen

def save_mutated_fcg(ori_fcg, ms, gen):
    fcg = copy.deepcopy(ori_fcg)
    fcg.process_mutations(ms)
    apk_name = fcg.apk_name
    save_path = os.path.join(save_dir, str(gen)+"_"+apk_name+".gexf")
    fcg.save_gexf(save_path)

def mutation_with_guidance_degree(fcg, target_bounds):
    original_feature = np.array(fcg.extract_feature(type=feature_type)).reshape(1, -1)

    add_sparse_nodes = 0
    add_dense_nodes = 0

    sensitive_nodes = fcg.sensitive_nodes
    longlink_length = [0] * len(sensitive_nodes)
    longlink_times = [0] * len(sensitive_nodes)

    # Step 1: Adjust degree feature (lowering degree by adding sparse nodes)
    # Step 2: Adjust longlink feature (increasing degree for specific nodes)

    total_nodes = len(fcg.nodes)

    # Adjust sparse nodes gene based on degree feature bounds
    for idx in range(len(target_bounds)):
        lower_bound, upper_bound = target_bounds[idx]
        if lower_bound == -1 and upper_bound == -1:
            continue

        actual_value = original_feature[0][idx]

        if actual_value > upper_bound:
            adjustment_ratio = actual_value / upper_bound
            add_sparse_nodes = max(add_sparse_nodes, int(adjustment_ratio * total_nodes))

    add_sparse_nodes = min(add_sparse_nodes, steps)

    # Create initial mutation state
    mutations = Mutations(add_sparse_nodes, add_dense_nodes, longlink_times, longlink_length, sensitive_nodes)

    # Calculate the new feature after applying mutations
    current_feature = calc_feature(fcg, mutations)

    # Adjust longlink gene if the actual value is lower than the bound
    for idx in range(len(target_bounds)):
        lower_bound, upper_bound = target_bounds[idx]
        if lower_bound == -1 and upper_bound == -1:
            continue

        actual_value = current_feature[0][idx]

        if actual_value < lower_bound:
            current_degree = fcg.unnorm_degree[idx]
            adjustment_ratio = lower_bound / actual_value
            node_id = fcg.sensitive_apis_bitmap[idx]
            gene_idx = sensitive_nodes.index(node_id)

            longlink_length[gene_idx] = 1
            if gene_idx not in longlink_times:
                longlink_times[gene_idx] = 1
            longlink_times[gene_idx] = max(int(adjustment_ratio * current_degree), longlink_times[gene_idx])
            longlink_times[gene_idx] = min(longlink_times[gene_idx], steps)

    # Apply the final mutation changes
    mutations = Mutations(add_sparse_nodes, add_dense_nodes, longlink_times, longlink_length, sensitive_nodes)

    # Calculate the new feature and check the attack success
    new_feature = calc_feature(fcg, mutations)
    attack_score = target_model.predict(new_feature).reshape(-1)

    # Check if the attack was successful
    if attack_score[0] < 0.5:
        return True, mutations, 0
    else:
        # If attack fails, continue with a genetic algorithm for further optimization
        success, mutations, generation = ga(
            fcg, target_bounds,
            add_sparse_nodes, add_dense_nodes,
            longlink_times, longlink_length, sensitive_nodes
        )
        return success, mutations, generation


def mutation_with_guidance_katz(fcg, target_att_bounds):
    alpha = 0.005

    # Extract original Katz feature
    ori_feature = np.array(fcg.extract_feature(type=feature_type)).reshape(1, -1)

    add_sparse_nodes_gene = 0
    add_density_nodes_gene = 0
    longlink_len_gene = [0] * len(fcg.sensitive_nodes)
    longlink_times_gene = [0] * len(fcg.sensitive_nodes)
    longlink_gene_detail = fcg.sensitive_nodes

    # Adjust Katz feature in two stages:
    # 1. First adjust by adding dense nodes
    # 2. Second adjust by increasing specific Katz features via longlink

    # Stage 1: Reduce Katz features by adding dense nodes
    for index, (lower_bound, upper_bound) in enumerate(target_att_bounds):
        if lower_bound == -1 and upper_bound == -1:
            continue
        actual_value = ori_feature[0][index]

        if actual_value > upper_bound:
            # Approximate dense node addition for Katz feature reduction
            add_density_nodes_gene = random.randint(steps, steps*10)

    add_density_nodes_gene = min(add_density_nodes_gene, steps*10)
    ms = Mutations(add_sparse_nodes_gene, add_density_nodes_gene, longlink_times_gene, longlink_len_gene,
                   longlink_gene_detail)

    # Calculate the current feature after mutation
    cur_feature = calc_feature(fcg, ms)

    # Stage 2: Increase specific Katz features by adjusting longlink
    for index, (lower_bound, upper_bound) in enumerate(target_att_bounds):
        if lower_bound == -1 and upper_bound == -1:
            continue
        actual_value = cur_feature[0][index]

        if actual_value < lower_bound:
            # Adjust longlink based on Katz propagation formula
            ori_degree = fcg.unnorm_degree[index]
            mid_result = lower_bound / actual_value * (1 + alpha * ori_degree)
            longlink_times = mid_result / alpha

            nodeid = fcg.sensitive_apis_bitmap[index]
            gene_index = longlink_gene_detail.index(nodeid)

            longlink_len_gene[gene_index] = 1
            longlink_times_gene[gene_index] = max(longlink_times, longlink_times_gene.get(gene_index, 1))
            longlink_times_gene[gene_index] = min(longlink_times_gene[gene_index], steps*10)

    # Perform mutation after adjustments
    ms = Mutations(add_sparse_nodes_gene, add_density_nodes_gene, longlink_times_gene, longlink_len_gene,
                   longlink_gene_detail)

    new_feature = calc_feature(fcg, ms)
    att_score = target_model.predict(new_feature).reshape(-1)

    if att_score[0] < 0.5:
        return True, ms, 0  # Attack succeeded

    # Enter GA if the attack fails after direct mutation
    result, ms, gen = ga(fcg, target_att_bounds, add_sparse_nodes_gene, add_density_nodes_gene,
                                                 longlink_times_gene, longlink_len_gene, longlink_gene_detail)

    return result, ms, gen

def mutation_with_guidance_harmonic(fcg, target_att_bounds):
    feature_type = 'harmonic'

    # Extract original harmonic feature
    ori_feature = np.array(fcg.extract_feature(type=feature_type)).reshape(1, -1)

    add_sparse_nodes_gene = 0
    add_density_nodes_gene = 0

    # Initialize longlink gene details (corresponding to sensitive nodes)
    longlink_gene_detail = fcg.sensitive_nodes
    longlink_len_gene = [0] * len(longlink_gene_detail)
    longlink_times_gene = [0] * len(longlink_gene_detail)

    # Harmonic feature adjustment: increase only, no decrease
    cur_feature = ori_feature
    for index, (lower_bound, upper_bound) in enumerate(target_att_bounds):
        if lower_bound == -1 and upper_bound == -1:
            continue
        actual_value = cur_feature[0][index]

        # If the harmonic feature needs to be increased, add longlinks
        if actual_value < lower_bound:
            # Calculate the gap and add longlink accordingly
            gap = lower_bound - actual_value
            longlink_times = int(gap) + 1  # Ensure at least one longlink is added
            node_id = fcg.sensitive_apis_bitmap[index]
            gene_index = longlink_gene_detail.index(node_id)

            longlink_len_gene[gene_index] = 1  # Set the length of longlink
            longlink_times_gene[gene_index] = max(longlink_times, longlink_times_gene.get(gene_index, 1))
            longlink_times_gene[gene_index] = min(longlink_times_gene[gene_index], 10000)  # Cap at 10,000

    # Create mutations object with the updated longlink information
    ms = Mutations(add_sparse_nodes_gene, add_density_nodes_gene, longlink_times_gene, longlink_len_gene, longlink_gene_detail)

    # Calculate the new feature after mutation
    new_feature = calc_feature(fcg, ms)
    att_score = target_model.predict(new_feature).reshape(-1)

    result = False
    if att_score[0] < 0.5:
        result = True
        gen = 0
    else:
        # Enter GA process if the attack fails
        result, ms, gen = ga(
            fcg, target_att_bounds,
            add_sparse_nodes_gene, add_density_nodes_gene,
            longlink_times_gene, longlink_len_gene, longlink_gene_detail
        )

    return result, ms, gen

def mutation_with_guidance_closeness(fcg, target_att_bounds):
    feature_type = 'closeness'
    ori_feature = np.array(fcg.extract_feature(type=feature_type)).reshape(1, -1)

    add_sparse_nodes_gene = 0
    add_density_nodes_gene = 0
    longlink_gene_detail = fcg.sensitive_nodes
    longlink_len_gene = [0] * len(longlink_gene_detail)
    longlink_times_gene = [0] * len(longlink_gene_detail)

    # Step 1: Reduce closeness feature (if needed)
    node_num = len(fcg.nodes)
    for index, (lower_bound, upper_bound) in enumerate(target_att_bounds):
        if lower_bound == -1 and upper_bound == -1:
            continue
        actual_value = ori_feature[0][index]

        # If closeness is greater than the upper bound, add sparse nodes
        if actual_value > upper_bound:
            ratio = actual_value / upper_bound
            add_sparse_nodes_gene = max(add_sparse_nodes_gene, int(ratio * node_num))

    ms = Mutations(add_sparse_nodes_gene, add_density_nodes_gene, longlink_times_gene, longlink_len_gene, longlink_gene_detail)
    cur_feature = calc_feature(fcg, ms)

    # Step 2: Increase closeness feature (if needed)
    for index, (lower_bound, upper_bound) in enumerate(target_att_bounds):
        if lower_bound == -1 and upper_bound == -1:
            continue
        actual_value = cur_feature[0][index]

        # If closeness is less than the lower bound, add long links
        if actual_value < lower_bound:
            neighbor_num, distance_sum = fcg.clossness_extra_details[index]
            lt1 = int(lower_bound / actual_value * (neighbor_num - 1) ** 2 / distance_sum - (neighbor_num - 1))
            lt2 = int((np.sqrt(10 * lower_bound / actual_value) - 1) * (neighbor_num - 1))
            longlink_times = max(lt1, lt2)

            node_id = fcg.sensitive_apis_bitmap[index]
            gene_index = longlink_gene_detail.index(node_id)
            longlink_len_gene[gene_index] = 1
            longlink_times_gene[gene_index] = max(longlink_times, longlink_times_gene.get(gene_index, 1))

            # Limit the long link times to 10000
            longlink_times_gene[gene_index] = min(longlink_times_gene[gene_index], 10000)

    ms = Mutations(add_sparse_nodes_gene, add_density_nodes_gene, longlink_times_gene, longlink_len_gene, longlink_gene_detail)
    new_feature = calc_feature(fcg, ms)
    att_score = target_model.predict(new_feature).reshape(-1)

    # If attack succeeds, return success
    if att_score[0] < 0.5:
        return True, ms, 0

    # If attack fails, enter GA for further optimization
    result, ms, gen = ga(
        fcg, target_att_bounds,
        add_sparse_nodes_gene, add_density_nodes_gene,
        longlink_times_gene, longlink_len_gene, longlink_gene_detail
    )
    return result, ms, gen

def mutation_with_guidance_average(fcg, target_att_bounds):

    # Extract original feature
    ori_feature = np.array(fcg.extract_feature(type=feature_type)).reshape(1, -1)

    add_sparse_nodes_gene = 0
    add_density_nodes_gene = 0

    # Initialize longlink gene details corresponding to sensitive nodes
    longlink_gene_detail = fcg.sensitive_nodes
    longlink_len_gene = [0] * len(longlink_gene_detail)
    longlink_times_gene = [0] * len(longlink_gene_detail)

    node_num = len(fcg.nodes)

    # First pass: Adjust average feature by increasing sparse nodes
    for index, (lower_bound, upper_bound) in enumerate(target_att_bounds):
        if lower_bound == -1 and upper_bound == -1:
            continue
        actual_value = ori_feature[0][index]

        # If the actual value exceeds the upper bound, reduce it by adding sparse nodes
        if actual_value > upper_bound:
            harmonic_feature = fcg.harmonic_feature[index]

            # Ensure harmonic feature does not exceed upper_bound * 4
            if harmonic_feature >= upper_bound * 4:
                continue

            # Calculate the ratio for adjustment
            ratio = (upper_bound * 4 - harmonic_feature) / (actual_value * 4 - harmonic_feature)
            add_sparse_nodes_gene = max(add_sparse_nodes_gene, int(ratio * node_num))
            add_density_nodes_gene = 800

    # Limit the number of sparse nodes added
    add_sparse_nodes_gene = min(add_sparse_nodes_gene, 200000)

    # Apply mutation to calculate the current feature
    ms = Mutations(add_sparse_nodes_gene, add_density_nodes_gene, longlink_times_gene, longlink_len_gene, longlink_gene_detail)
    cur_feature = calc_feature(fcg, ms)

    # Second pass: Adjust average feature by increasing longlinks
    for index, (lower_bound, upper_bound) in enumerate(target_att_bounds):
        if lower_bound == -1 and upper_bound == -1:
            continue
        actual_value = cur_feature[0][index]

        # If actual value is below the lower bound, increase longlinks
        if actual_value < lower_bound:
            cur_degree = fcg.unnorm_degree[index]
            gap = lower_bound - actual_value
            longlink_times = int(gap / 0.25) + 1  # Each longlink increases by 0.25
            node_id = fcg.sensitive_apis_bitmap[index]
            gene_index = longlink_gene_detail.index(node_id)

            longlink_len_gene[gene_index] = 1
            longlink_times_gene[gene_index] = max(longlink_times, longlink_times_gene.get(gene_index, 1))
            longlink_times_gene[gene_index] = min(longlink_times_gene[gene_index], 10000)  # Cap at 10,000

    # Create mutation object with updated longlink information
    ms = Mutations(add_sparse_nodes_gene, add_density_nodes_gene, longlink_times_gene, longlink_len_gene, longlink_gene_detail)
    new_feature = calc_feature(fcg, ms)
    att_score = target_model.predict(new_feature).reshape(-1)

    if att_score[0] < 0.5:
        result = True
        gen = 0
    else:
        # If attack fails, enter Genetic Algorithm (GA) for further optimization
        result, ms, gen = ga(
            fcg, target_att_bounds,
            add_sparse_nodes_gene, add_density_nodes_gene, 
            longlink_times_gene, longlink_len_gene, longlink_gene_detail
        )

    return result, ms, gen


def mutation_with_guidance_combine(fcg, target_att_bounds):
    alpha = 0.005
    ori_feature = np.array(fcg.extract_feature(type=feature_type)).reshape(1, -1)

    # Three mutation types:
    # 1. Add sparse nodes: Add n1 new nodes and edges from the root_node to new nodes.
    #    root_node is chosen from fcg.reserve_user_node.
    #    n1 = add_sparse_nodes_gene
    # 2. Add dense nodes: Add n2 new nodes and edges from root_node to new nodes, then connect lower index nodes
    #    to higher index nodes.
    #    n2 = add_density_nodes_gene
    # 3. Add long links: Add t long links with length l, pointing to node N. The start node is root_node, and
    #    l connected nodes are added sequentially, with the last node linking to N. Repeat t times.
    #    t = longlink_times_gene[i], l = longlink_len_gene[i], N = longlink_gene_detail[i]

    add_sparse_nodes_gene = 0
    add_density_nodes_gene = 0
    longlink_gene_detail = fcg.sensitive_nodes
    longlink_len_gene = [0] * len(longlink_gene_detail)
    longlink_times_gene = [0] * len(longlink_gene_detail)

    # Step 1: Reduce feature values (if needed)
    for index, (lower_bound, upper_bound) in enumerate(target_att_bounds):
        if lower_bound == -1 and upper_bound == -1:
            continue
        actual_value = ori_feature[0][index]

        # For degree feature (0-430), reduce by adding sparse nodes
        if index < 430 and actual_value > upper_bound:
            ratio = actual_value / upper_bound
            node_num = len(fcg.nodes)
            add_sparse_nodes_gene = max(add_sparse_nodes_gene, int(ratio * node_num))

        # For Katz feature (430-860), reduce by adding dense nodes
        elif index < 860 and actual_value > upper_bound:
            add_density_nodes_gene = min(800, 100 * (int(len(fcg.nodes) / 100) + 1))

        # For Closeness feature (860-1290), reduce by adding sparse nodes
        elif index < 1290 and actual_value > upper_bound:
            ratio = actual_value / upper_bound
            node_num = len(fcg.nodes)
            add_sparse_nodes_gene = max(add_sparse_nodes_gene, int(ratio * node_num))

    add_sparse_nodes_gene = min(add_sparse_nodes_gene, 200000)

    # Apply initial mutations to reduce features
    ms = Mutations(add_sparse_nodes_gene, add_density_nodes_gene, longlink_times_gene, longlink_len_gene,
                   longlink_gene_detail)
    cur_feature = calc_feature(fcg, ms)

    # Step 2: Increase feature values (if needed)
    for index, (lower_bound, upper_bound) in enumerate(target_att_bounds):
        if lower_bound == -1 and upper_bound == -1:
            continue
        actual_value = cur_feature[0][index]

        # Increase degree feature (0-430) by adding long links
        if index < 430 and actual_value < lower_bound:
            cur_degree = fcg.unnorm_degree[index]
            ratio = lower_bound / actual_value
            node_id = fcg.sensitive_apis_bitmap[index]
            gene_index = longlink_gene_detail.index(node_id)
            longlink_len_gene[gene_index] = 1
            longlink_times_gene[gene_index] = max(int(ratio * cur_degree), longlink_times_gene[gene_index])

        # Increase Katz feature (430-860) by adding long links
        elif index < 860 and actual_value < lower_bound:
            ori_degree = fcg.unnorm_degree[index - 430]
            mid_result = lower_bound / actual_value * (1 + alpha * ori_degree)
            longlink_times = int(mid_result / alpha)
            node_id = fcg.sensitive_apis_bitmap[index - 430]
            gene_index = longlink_gene_detail.index(node_id)
            longlink_len_gene[gene_index] = 1
            longlink_times_gene[gene_index] = max(longlink_times, longlink_times_gene.get(gene_index, 1))

        # Increase Closeness feature (860-1290) by adding long links
        elif index < 1290 and actual_value < lower_bound:
            neighbor_num, distance_sum = fcg.closeness_extra_details[index - 860]
            lt1 = int(lower_bound / actual_value * (neighbor_num - 1) ** 2 / distance_sum - (neighbor_num - 1))
            lt2 = int((np.sqrt(10 * lower_bound / actual_value) - 1) * (neighbor_num - 1))
            longlink_times = max(lt1, lt2)
            node_id = fcg.sensitive_apis_bitmap[index - 860]
            gene_index = longlink_gene_detail.index(node_id)
            longlink_len_gene[gene_index] = 1
            longlink_times_gene[gene_index] = max(longlink_times, longlink_times_gene.get(gene_index, 1))

        # Increase Harmonic feature (1290-1720) by adding long links
        elif index >= 1290 and actual_value < lower_bound:
            gap = lower_bound - actual_value
            longlink_times = int(gap) + 1
            node_id = fcg.sensitive_apis_bitmap[index - 1290]
            gene_index = longlink_gene_detail.index(node_id)
            longlink_len_gene[gene_index] = 1
            longlink_times_gene[gene_index] = max(longlink_times, longlink_times_gene.get(gene_index, 1))

    # Cap longlink_times_gene to 1000
    longlink_times_gene = [min(times, 1000) for times in longlink_times_gene]

    # Apply final mutations and calculate new features
    ms = Mutations(add_sparse_nodes_gene, add_density_nodes_gene, longlink_times_gene, longlink_len_gene,
                   longlink_gene_detail)
    new_feature = calc_feature(fcg, ms)
    att_score = target_model.predict(new_feature).reshape(-1)

    if att_score[0] < 0.5:
        return True, ms, 0
    else:
        # Enter GA for further optimization if attack fails
        return ga(fcg, target_att_bounds,
                                          add_sparse_nodes_gene, add_density_nodes_gene,
                                          longlink_times_gene, longlink_len_gene, longlink_gene_detail)


def calc_feature(ori_fcg, ms):
    fcg = copy.deepcopy(ori_fcg)
    fcg.process_mutations(ms)

    feature = fcg.extract_feature(type=feature_type)
    return np.array(feature).reshape(1, -1)

def parse_arguments():
    """
    Function to initialize the argument parser and parse command-line arguments.

    Returns:
    --------
    args : argparse.Namespace
        Parsed arguments from the command line.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process some input paths and settings.")

    # Define arguments
    parser.add_argument('--save_path', type=str, required=True, help="Directory to save the results.")
    parser.add_argument('--attack_sample', type=str, required=True, help="Path to the attack samples.")
    parser.add_argument('--target_model', type=str, required=True, help="Path to the target model.")
    parser.add_argument('--feature_type', type=str,
                        choices=['concentrate', 'average', 'degree', 'katz', 'closeness', 'harmonic'], required=True,
                        help="Type of MalScan feature extraction.")


    # basic parameters
    parser.add_argument('--pop_num', type=int, required=True, help="Number of population during GA.")
    parser.add_argument('--max_generation', type=int, required=True, help="Number of max generations.")
    parser.add_argument('--steps', type=int, required=True, help="Number of mutation steps.")

    return parser.parse_args()

if __name__ == '__main__':
    # Systematically set the CPU affinity for the current process
    p = psutil.Process(os.getpid())
    p.cpu_affinity(list(range(1, 110)))

    # For running the script in parallel
    if len(sys.argv) > 2:
        try:
            proc_id = int(sys.argv[1])
            proc_total = int(sys.argv[2])
            print(f"Received part index: {proc_id}")
        except ValueError:
            proc_id = 0
            proc_total = 1
            print("Error: The first argument must be an integer.")
    else:
        proc_id = 0
        proc_total = 1

    # 1.Initialize the parameters
    args = parse_arguments()

    # 2. Read attack samples from txt file
    attack_samples = glob.glob(args.attack_sample)
    print("attack_samples", len(attack_samples))

    part_size = len(attack_samples) // proc_total
    extra = len(attack_samples) % proc_total
    start_index = proc_id * part_size + min(proc_id, extra)
    if proc_id < extra:
        part_size += 1
    end_index = start_index + part_size

    # load model, global variable
    target_model = load_model(args.target_model)

    feature_type = args.feature_type

    for i in range(start_index, end_index):
        fcg_file = attack_samples[i]
        fcg = FCG(fcg_file, label=1)
        combined_feature = fcg.extract_feature(type=feature_type)
        if combined_feature is not None:
            combined_feature = np.array(combined_feature)
            combined_feature = combined_feature.reshape(1, -1)

            score = target_model.predict_proba(combined_feature)

            if score[0][1] < 0.5:
                print("benign")

            else:
                print("malware")

                save_dir = args.save_path
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                generation = args.max_generation
                pop_num = args.pop_num
                steps = args.steps

                # prepare target bounds
                fcg.reduce()
                attackable, not_sure, target_att_bounds =process_tree_based_model(fcg, target_model, combined_feature, feature_type)

                if not attackable:
                    print("not attackable malware")
                    continue

                else:
                    # first mutation, next enter ga
                    att_success, gen = mutation_with_guidance(fcg, target_att_bounds)





