import sys
import os
import pygmo as pg
import tensorflow as tf
import torch
import argparse
import random
import copy
import psutil
import glob
import time


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from type.FCG_mamadroid import FCG_mamadroid
import numpy as np
from utils import load_model, load_GCN_model, load_MLP_model
from utils import save_graph_for_every_generation, save_log_for_every_generation, save_log_for_every_individual
from utils import test_GCN_model, obtain_gcn_feature
from utils import _init_population
from utils import deal_with_conflict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)


#global variable
save_dir = None
model = None #surrogate model, GPU
sub_model = None #surrogate model
sub_model_name = None
target_model = None #target model
target_model_name = None



def ga(fcg, max_generation=100, pop_num=100, steps=300):
    """
        Genetic Algorithm (GA) for optimizing the function call graph (FCG) through mutation and crossover operations.
        Key steps:
        1. **Initialization**: A population of FCG mutations is created with `_init_population`.
        2. **Fitness Evaluation**: The population is scored using `_population_score` to evaluate how well each individual performs.
        3. **Selection**: The two best-performing individuals are selected using `_ga_select` for crossover and mutation.
        4. **Crossover & Mutation**: New individuals are generated by crossing over two selected parents and applying mutation steps to further evolve them.
        5. **Conflict Resolution**: Any conflicts within the mutated FCGs are resolved using `deal_with_conflict`.
        6. **Evolution**: The top-performing individuals from the current and previous generations are selected to form the next generation.
        7. **Termination**: The process repeats for `max_generation` iterations, or stops early if an optimal solution is found.
    """

    save_state = False

    population = _init_population(fcg, pop_num, steps)
    pop_score_list, pop_shap_list = _population_score(fcg, population)
    indexes = np.argsort(pop_score_list)[:pop_num]
    population = [population[i] for i in indexes]
    pop_score_list = [pop_score_list[i] for i in indexes]
    pop_shap_list = [pop_shap_list[i] for i in indexes]
    if len(pop_score_list) == 0:
        return None


    for i in range(max_generation):
        tf.keras.backend.clear_session()

        target_index = _fitness_function(pop_score_list)
        if target_index != -1:
            print("fitness function is true")
            save_log_for_every_individual(fcg, population[target_index], pop_score_list[target_index], i, save_dir)
            break

        next_gene_population = []
        save_log_for_every_generation(fcg, pop_score_list, pop_shap_list, i, population, save_dir)

        while True:
            if len(next_gene_population) >= pop_num:
                next_gene_population = next_gene_population[:pop_num]
                break

            individual1_index, individual2_index = _ga_select(population, pop_score_list, pop_shap_list, int(pop_num / 2) - 1)
            individual1 = population[individual1_index]
            individual2 = population[individual2_index]

            new_individual = copy.deepcopy(individual1)
            new_individual.crossover(individual2, fcg)
            cur_fcg = deal_with_conflict(fcg, new_individual)
            new_individual.ga_mutation_add_edge(cur_fcg, 0.2, False)
            new_individual.update_dependency()
            next_gene_population.append(new_individual)

        if i == max_generation - 1:
        # last generation, saving all the graphs for analysis
            save_state = True
        cur_pop_score_list, cur_pop_shap_list = _population_score(fcg, next_gene_population, save_state)
        if len(cur_pop_score_list) == 0:
            return None
        combined_population = population + next_gene_population
        combined_score_list = pop_score_list + cur_pop_score_list
        combined_shap_list = pop_shap_list + cur_pop_shap_list

        top_pop, top_score, top_shap = sort_all_pop(combined_population, combined_score_list, combined_shap_list, pop_num)
        population = top_pop
        pop_score_list = top_score
        pop_shap_list = top_shap

    return population


def _individual_score(fcg, individual):
    fcg = copy.deepcopy(fcg)

    original_combined_feature = extract_feature_use_fcg(fcg)
    if original_combined_feature is None:
        print("feature is none")
        return None

    for j in range(len(individual.final_group_list)):
        group = individual.final_group_list[j]
        safe_mutation = []
        for i in range(len(group)):
            mutation = group[i]
            state, res = fcg.process_mutation(mutation)
            if state:
                safe_mutation.append(mutation)
        individual.final_group_list[j] = safe_mutation

    # 1. features
    combined_feature = extract_feature_use_fcg(fcg)
    if combined_feature is None:
        print("new feature is none")
        return None

    return original_combined_feature, combined_feature, fcg



def extract_feature_use_fcg(fcg):
    feature = fcg.cal_mamadroid_feature()
    non_zero_feature = np.count_nonzero(feature)
    if non_zero_feature != 0:
        feature = feature.flatten()
        return feature
    else:
        return None


def _fitness_function(pop_score_list):
    target_index = -1
    for i in range(len(pop_score_list)):
        if pop_score_list[i] is not None and pop_score_list[i] < 0.5:
            target_index = i
            break
    return target_index


def _population_score(fcg, pop, save_state=False):
    # original score + sub score
    score_list = []
    sub_score_list = []
    pop_feature_list = []
    pop_fcg_list = []

    if sub_model_name == 'GCN':
        for i in range(len(pop) - 1, -1, -1):

            res = _individual_score(fcg, pop[i])
            # if score is None or shap_sum is None:
            if res is None:
                # if score is None, remove the individual
                del pop[i]
            else:
                original_combined_feature, combined_feature, cur_fcg = res
                data = obtain_gcn_feature(cur_fcg)
                sub_score = test_GCN_model(data, sub_model)

                sub_score_list.append(sub_score)
                combined_feature = combined_feature.reshape(1, -1)
                pop_feature_list.extend(combined_feature)
                pop_fcg_list.append(cur_fcg)

        pop_feature_list = np.array(pop_feature_list)
        all_pop_score = target_model.predict_proba(pop_feature_list)
        for i in range(len(all_pop_score)):
            score = all_pop_score[i][1]
            score_list.append(score)

            cur_fcg = pop_fcg_list[i]
            if score < 0.5:
                save_graph_for_every_generation(cur_fcg, 'success', -1, save_dir)


    elif sub_model_name == 'MLP':
        for i in range(len(pop) - 1, -1, -1):
            res = _individual_score(fcg, pop[i])
            # if score is None or shap_sum is None:
            if res is None:
                del pop[i]
            else:
                original_combined_feature, combined_feature, cur_fcg = res
                combined_feature = combined_feature.reshape(1, -1)
                pop_feature_list.extend(combined_feature)
                pop_fcg_list.append(cur_fcg)

        pop_feature_list = np.array(pop_feature_list)
        all_pop_score = target_model.predict(pop_feature_list)
        all_sub_score = sub_model.predict(pop_feature_list)
        for i in range(len(all_pop_score)):
            score = all_pop_score[i]
            score_list.append(score)
            sub_score_list.append(-all_sub_score[i][0])  # because the gcn score represent lager is good

            cur_fcg = pop_fcg_list[i]
            if score < 0.5:
                save_graph_for_every_generation(cur_fcg, 'success', -1, save_dir)


    return score_list[::-1], sub_score_list[::-1]


def select(pop_score_list, pop_shap_list, idxes=None):
    # dominate
    pop_score_candidate = [pop_score_list[idx] for idx in idxes]
    pop_shap_candidate = [pop_shap_list[idx] for idx in idxes]

    if len(idxes) == 1:
        return idxes[0]

    # For the score, the smaller the better; for the shap, the larger the better
    com = []
    for i in range(len(pop_score_candidate)):
        score = pop_score_candidate[i]
        shap = -pop_shap_candidate[i]
        com.append([score, shap])

    # print("com", com)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=com)

    return idxes[ndf[0][0]]


def sort_all_pop(pop, pop_score_list, pop_shap_list, pop_num):
    # dominate
    com = []
    for i in range(len(pop_score_list)):
        score = pop_score_list[i]
        shap = -pop_shap_list[i]
        com.append([score, shap])

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=com)
    pop_total = []
    pop_score_list_total = []
    pop_shap_list_total = []
    for level in ndf:
        for idx in level:
            # print("idx", idx)
            pop_total.append(pop[idx])
            pop_score_list_total.append(pop_score_list[idx])
            pop_shap_list_total.append(pop_shap_list[idx])
            if len(pop_total) >= pop_num:
                break

        if len(pop_total) >= pop_num:
            break

    return pop_total, pop_score_list_total, pop_shap_list_total


def _ga_select(pop, pop_score_list, pop_shap_list, tournament_size=20):
    # Ensure the tournament size does not exceed the population size
    tournament_size = min(tournament_size, len(pop))

    # Select two individuals' indices
    winner_indices = set()
    while True:
        # Randomly select tournament size individuals
        tournament_indices = random.sample(range(len(pop)), tournament_size)
        # Comprehensively consider the score and shap of the selected individuals
        winner_index = select(pop_score_list, pop_shap_list, tournament_indices)
        # Adds to the list of winner subscripts
        winner_indices.add(winner_index)
        if len(winner_indices) >= 2:
            break

    winner_indices = list(winner_indices)
    return winner_indices[0], winner_indices[1]



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
    parser.add_argument('--target_model_name', type=str, required=True, help="Name of to the target model.")
    parser.add_argument('--surrogate_model', type=str, required=True, help="Path to the surrogate model.")
    parser.add_argument('--surrogate_model_name', type=str, choices=['MLP', 'GCN'], required=True,
                        help="Name of the surrogate model (must be MLP or GCN).")

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

    if args.surrogate_model_name == 'MLP':
        sub_model = load_MLP_model(args.surrogate_model)
    elif args.surrogate_model_name == 'GCN':
        sub_model = load_GCN_model(args.surrogate_model)
    print("sub_model", sub_model)

    sub_model_name = args.surrogate_model_name

    # load model, global variable
    target_model = load_model(args.target_model)

    # generate the new shap for the failed samples
    all_samples_features = []

    zeros_list = [0] * 121

    for i in range(start_index, end_index):
        fcg_file = attack_samples[i]

        load_start = time.time()
        fcg = FCG_mamadroid(fcg_file, 1, zeros_list)

        feature = fcg.cal_mamadroid_feature()
        non_zero_feature = np.count_nonzero(feature)
        if non_zero_feature != 0:
            feature = feature.flatten()
            all_samples_features.append(feature)

            new_feature = feature.reshape(1, -1)
            Y_probs = target_model.predict_proba(new_feature)
            print("Y_probs", Y_probs)

            if sub_model_name == 'MLP':
                sub_score = sub_model.predict(new_feature)
            else:
                data = obtain_gcn_feature(fcg)
                sub_score = test_GCN_model(data, sub_model)
            print("sub_score", sub_score)

            if Y_probs[0][1] < 0.5:
                print("benign")

            else:
                print("malware")

                save_dir = args.save_path
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                generation = args.max_generation
                pop_num = args.pop_num
                steps = args.steps

                ga(fcg, generation, pop_num, steps)




