import sys
import os
import torch
import argparse
import random
import copy
import psutil
import glob


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from type.FCG_ma_api_tree import FCG_ma_api_tree as FCG
from utils import load_model
from utils import process_tree_based_model_mamadroid_apigraph as process_tree_based_model
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


def ga(fcg, target_att_bounds):
    """
    Genetic Algorithm for optimizing Mamadroid features by adjusting the number of added edges.

    Parameters:
    - fcg: The function call graph.
    - target_att_bounds: Bounds for features that need to be adjusted.

    Returns:
    - result: Whether a successful attack was found.
    - best_individual: The best individual (mutation) found by the GA.
    - generation_count: The number of generations taken to find the solution.
    """

    # Initialize population and set random seed
    random.seed(42)
    init_population_size = pop_num * 10
    population_size = pop_num
    population = []

    types_num = len(fcg.type_count)
    max_add_num = np.zeros((types_num, types_num), dtype=int)

    # Calculate the maximum number of edges that can be added between each pair of node types
    for i in range(types_num):
        for j in range(types_num):
            max_add_num[i][j] = int(fcg.type_count[i] * fcg.type_count[j] - fcg.detail_edges[i][j])

    # Create the initial population
    for _ in range(init_population_size):
        add_num_individual = np.zeros((types_num, types_num))
        for i in range(types_num - 2, types_num):
            for j in range(types_num):
                tmp_max_add_num = min(steps, max_add_num[i][j])
                add_num_individual[i][j] = random.randint(0, tmp_max_add_num)
        population.append(add_num_individual)

    generation = max_generation

    for curr_gen in range(generation):
        # Calculate fitness for each individual in the population
        fitness = []
        for individual in population:
            feature = calc_feature(fcg, individual)
            att_score = target_model.predict(feature).reshape(-1)

            # Check if attack is successful
            if att_score[0] < 0.5:
                save_mutated_fcg(fcg, individual, curr_gen)
                return True, individual, curr_gen + 1  # Successful attack

            # Calculate fitness based on how many features are within the target bounds
            fit_score = 0
            for index in range(len(target_att_bounds)):
                lower_bound, upper_bound = target_att_bounds[index]
                if lower_bound != -1 and upper_bound != -1:
                    actual_value = feature[0][index]
                    if lower_bound < actual_value < upper_bound:
                        fit_score += 1
            fitness.append(fit_score)

        # Sort population by fitness in descending order
        population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0], reverse=True)]

        # Generate the next generation
        next_generation = []
        while len(next_generation) < population_size:
            # Select 2 individuals using tournament selection
            group_size = int(population_size * 0.2)
            group1 = random.sample(range(population_size), group_size)
            group2 = random.sample(range(population_size), group_size)

            best1 = population[group1[0]]
            best2 = population[group2[0]]

            # Crossover to produce a new individual
            new_individual = np.zeros((types_num, types_num))
            for i in range(types_num):
                for j in range(types_num):
                    new_individual[i][j] = random.choice([best1[i][j], best2[i][j]])

            # Mutation with a 5% chance
            if random.random() < 0.05:
                for i in range(types_num - 2, types_num):
                    for j in range(types_num):
                        new_individual[i][j] = random.randint(0, max_add_num[i][j])

            next_generation.append(new_individual)

        # Update population with the new generation
        population = next_generation

    return False, None, generation  # No successful attack found after all generations


def mutation_with_guidance(fcg, target_att_bounds):
    # Helper function to convert feature index to type ('self-defined', 'obfuscated', etc.)
    def index_to_type(index):
        types_num = len(fcg.type_count)
        if index < (types_num - 2) * types_num:
            return "NA"
        elif index < (types_num - 1) * types_num:
            return "self-defined"
        else:
            return "obfuscated"

    # Helper function to map feature index to a pair of types
    def featureindex_to_typepair(index):
        types_num = len(fcg.type_count)
        from_type = index // types_num
        to_type = index % types_num
        return from_type, to_type

    ori_feature = fcg.extract_features(feature_type)

    # Identify mutable indexes, representing edges that can be added
    mutable_indexes = []
    types_num = len(fcg.type_count)

    # Check for self-defined edges
    if fcg.type_count[types_num - 2] > 0:
        for i in range(types_num):
            if fcg.type_count[i] > 0:
                index = (types_num - 2) * types_num + i
                mutable_indexes.append(index)

    # Check for obfuscated edges
    if fcg.type_count[types_num - 1] > 0:
        for i in range(types_num):
            if fcg.type_count[i] > 0:
                index = (types_num - 1) * types_num + i
                mutable_indexes.append(index)

    # Identify critical features based on bounds
    critical_feature_index = []
    for index in range(len(target_att_bounds)):
        lower_bound, upper_bound = target_att_bounds[index]
        if lower_bound != -1 and upper_bound != -1:
            critical_feature_index.append(index)


    # Calculate required number of edges to decrease certain features
    self_defined_addnum, obfuscated_addnum = 0, 0
    for index in critical_feature_index:
        lower_bound, upper_bound = target_att_bounds[index]
        actual_value = ori_feature[index]
        if actual_value > upper_bound:
            from_type, to_type = featureindex_to_typepair(index)
            ori_edge_num = fcg.detail_edges[from_type][to_type]
            ratio = actual_value / upper_bound
            addnum = int(ratio * ori_edge_num)
            if index_to_type(index) == "self-defined":
                self_defined_addnum = max(self_defined_addnum, addnum)
            else:
                obfuscated_addnum = max(obfuscated_addnum, addnum)

    # Prepare matrix to store the number of additional edges needed for mutation
    add_num_detail = np.zeros((types_num, types_num))

    # Try to add edges for self-defined types
    self_defined_addnum_needed = self_defined_addnum
    for i in range(types_num):
        if self_defined_addnum_needed <= 0:
            break
        max_add_num = fcg.type_count[i] * fcg.type_count[types_num - 2] - fcg.detail_edges[types_num - 2][i]
        if max_add_num > self_defined_addnum_needed:
            add_num_detail[types_num - 2][i] = self_defined_addnum_needed
            self_defined_addnum_needed = 0
        else:
            add_num_detail[types_num - 2][i] = max_add_num
            self_defined_addnum_needed -= max_add_num

    # Try to add edges for obfuscated types
    obfuscated_addnum_needed = obfuscated_addnum
    for i in range(types_num):
        if obfuscated_addnum_needed <= 0:
            break
        max_add_num = fcg.type_count[i] * fcg.type_count[types_num - 1] - fcg.detail_edges[types_num - 1][i]
        if max_add_num > obfuscated_addnum_needed:
            add_num_detail[types_num - 1][i] = obfuscated_addnum_needed
            obfuscated_addnum_needed = 0
        else:
            add_num_detail[types_num - 1][i] = max_add_num
            obfuscated_addnum_needed -= max_add_num

    # Apply mutations to the edge matrix
    detail_new = copy.deepcopy(fcg.detail_edges)
    for i in range(types_num):
        for j in range(types_num):
            detail_new[i][j] += add_num_detail[i][j]

    # Calculate new features after mutation
    new_feature = fcg._normalizing_matrix(detail_new).flatten()

    # Check if additional edges are needed to meet the lower bound
    for index in critical_feature_index:
        lower_bound, upper_bound = target_att_bounds[index]
        actual_value = new_feature[index]
        if actual_value < lower_bound:
            from_type, to_type = featureindex_to_typepair(index)
            ori_edge_num = fcg.detail_edges[from_type][to_type]
            ratio = lower_bound / actual_value if actual_value != 0 else lower_bound
            addnum = int(ratio * ori_edge_num)
            add_num_detail[from_type][to_type] += addnum

    # Validate and limit the number of edges added
    for i in range(types_num - 2, types_num):
        for j in range(types_num):
            max_add_num = fcg.type_count[j] * fcg.type_count[i] - fcg.detail_edges[i][j]
            add_num_detail[i][j] = min(add_num_detail[i][j], max_add_num)

    # Final feature calculation after mutation
    final_feature = fcg._normalizing_matrix(copy.deepcopy(fcg.detail_edges)).flatten().reshape(1, -1)

    # Predict attack result using the model
    att_score = target_model.predict(final_feature).reshape(-1)
    if att_score[0] < 0.5:
        save_mutated_fcg(fcg, add_num_detail, 0)
        return True, 0, len(fcg.edges)  # Successful attack in initial mutation

    # Perform genetic algorithm (GA) if the initial attack failed
    result, add_num_detail, gen = ga(fcg, target_att_bounds)

    if result:
        ori_edge_num = len(fcg.edges)
        new_edge_num = np.sum(add_num_detail[types_num - 2: types_num])
        perturbation = new_edge_num / (ori_edge_num + 1e-10)
    else:
        perturbation = 0

    return result, gen, perturbation

def save_mutated_fcg(ori_fcg, ms, gen):
    fcg = copy.deepcopy(ori_fcg)
    fcg.process_mutations(ms)
    apk_name = fcg.apk_name
    save_path = os.path.join(save_dir, str(gen)+"_"+apk_name+".gexf")
    fcg.save_gexf(save_path)

def calc_feature(fcg, add_num_detail):
    detail_new = copy.deepcopy(fcg.detail_edges)
    types_num = len(fcg.type_count)
    for i in range(types_num):
        for j in range(types_num):
            detail_new[i][j] += add_num_detail[i][j]

    final_feature = fcg._normalizing_matrix(detail_new)
    final_feature = final_feature.flatten()
    final_feature = np.array(final_feature).reshape(1, -1)
    return final_feature

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
                        choices=['mamadroid', 'apigraph'], required=True,
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
        combined_feature = fcg.extract_features(feature_type)
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
                attackable, not_sure, target_att_bounds =process_tree_based_model(fcg, target_model, combined_feature)

                if not attackable:
                    print("not attackable malware")
                    continue

                else:
                    # first mutation, next enter ga
                    att_success, gen = mutation_with_guidance(fcg, target_att_bounds)





