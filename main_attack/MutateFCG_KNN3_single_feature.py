#March 8, attacking KNN using MLP substitue model, dont modify
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import os
sys.path.append(os.path.abspath('../type'))
sys.path.append(os.path.abspath('task'))

import tensorflow as tf
import pickle
import pygmo as pg
import networkx as nx
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 为每个 GPU 设置内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from FCG import FCG
import random
import numpy as np
from Mutation import Mutation
from MutationSequence import MutationSequence
from Classify import train_model, test_model,get_distance_from_knn1, get_distance_from_knn5, get_4features_data, get_distance_from_original_knn1
import copy
import json
import psutil
import glob
import time
from pympler import asizeof
from scipy.special import softmax
import joblib
from multiprocessing import Pool as ThreadPool
from functools import partial
import glob
import csv
from testShap import obtain_dataset, load_model, get_shap
import numpy as np
import pandas as pd

import psutil
import os

# 获取当前进程
p = psutil.Process(os.getpid())

# 设置希望进程运行的CPU列表，这里是从11到119号CPU
cpus = list(range(1, 40))

# 设置CPU亲和性
p.cpu_affinity(cpus)

# 打印当前进程的CPU亲和性设置，确认是否设置成功
print("Current CPU Affinity: ", p.cpu_affinity())


#individual是一个mutation_sequence, 是二维的list

#define a goble model
#load model
model = None
MLP_model = None
recover_path = ''#从哪个文件夹里面恢复
shap_path =''
init_num = 100
feature_dir = '/data/b/shiwensong/dataset/feature_Nov30/'
model_name = 'knn_3'
type = 'degree'
save_dir = f'./{type}_test_mar22_third/{model_name}/'#最后save的文件夹
# save_dir = 'test'
recover_dir = f'./{type}_test_mar22_second/{model_name}/'#从哪个文件夹里面恢复
benign_knn_model = None
malware_knn_model = None



#2023 2022 2019 can use

def random_m(fcg, steps, sensitive):
    # print("random_m starting")
    fcg = copy.deepcopy(fcg)
    #将给定的fcgs随机变换若干步,并保存
    # 获取一个 apk 的 call graph
    # print("fcg nodes", len(fcg.nodes))
    # print("fcg edges", len(fcg.edges))

    #每次传入是一个新的fcg

    # print("steps", steps)

    mutation_list = []
    visited_random_number = set()

    cnt_random_1 = 0 #add node 1
    cnt_random_2 = 0  # add node 2
    cnt_random_3 = 0 #add edge 1
    cnt_random_4 = 0 #add edge 2
    # cnt_random_5 = 0 # insert node
    # cnt_random_5 = 0 #add edge 3
    # cnt_random_6 = 0 #add edge 4
    # cnt_random_7 = 0 #rewiring 1
    # cnt_random_8 = 0  # rewiring 2
    # cnt_random_9 = 0 #remove node
    # cnt_random_10 = 0#remove node

    for i in range(100):
        mutation = fcg.build_mutation_add_node()
        state, res = fcg.process_mutation(mutation)
        if state:
            mutation_list.append(mutation)

    i = 0
    while i < steps:
        #生成随机数1-4
        # print('iteration', i)
        # print("mutation ing edges", len(mutation.edges), len(fcg.edges))
        # print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))
        random_int = random.randint(1, 4)
        # random_int = 2
        # print("random_int", random_int)

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
            mutation = fcg.build_mutation_remove_node()
            cnt_random_4 += 1
        # elif random_int == 5:
        #     mutation = fcg.build_mutation_insert_node()
        #     cnt_random_5 += 1
        # elif random_int == 6:
        #     mutation = fcg.build_mutation_add_edge_type4()
        #     cnt_random_6 += 1
        # elif random_int == 7:
        #     mutation = fcg.build_mutation_rewiring_type1()
        #     cnt_random_7 += 1
        # elif random_int == 8:
        #     mutation = fcg.build_mutation_rewiring_type2()
        #     cnt_random_8 += 1
        # elif random_int == 9:
        #     mutation = fcg.build_mutation_remove_node_type1()
        #     cnt_random_9 += 1
        # elif random_int == 10:
        #     mutation = fcg.build_mutation_remove_node_type2()
        #     cnt_random_10 += 1

        # print(mutation.mutation)
        if mutation is None:
            # print("mutation failed")
            if random_int == 1:
                cnt_random_1 -= 1
            elif random_int == 2:
                cnt_random_2 -= 1
            elif random_int == 3:
                cnt_random_3 -= 1
            elif random_int == 4:
                cnt_random_4 -= 1
            # elif random_int == 5:
            #     cnt_random_5 -= 1
            # elif random_int == 6:
            #     cnt_random_6 -= 1
            # elif random_int == 7:
            #     cnt_random_7 -= 1
            # elif random_int == 8:
            #     cnt_random_8 -= 1
            # elif random_int == 9:
            #     cnt_random_9 -= 1
            # elif random_int == 10:
            #     cnt_random_10 -= 1

            visited_random_number.add(random_int)
            # if visited_random_number == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}:
            if visited_random_number == {1, 2, 3, 4}:
                print("all mutations failed")
                break

        else:
            # print("mutation", mutation)
            state, res = fcg.process_mutation(mutation)
            if state:
                mutation_list.append(mutation)
            else:
                continue
            
        i += 1
    # print("mutation ing edges", len(mutation.edges), len(fcg.edges))
    # print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))

    # print("cnt_random_1", cnt_random_1)
    # print("cnt_random_2", cnt_random_2)
    # print("cnt_random_3", cnt_random_3)
    # print("cnt_random_4", cnt_random_4)


    # print("cnt_random_5", cnt_random_5)
    # print("cnt_random_6", cnt_random_6)
    # print("cnt_random_7", cnt_random_7)
    # print("cnt_random_8", cnt_random_8)
    # print("cnt_random_9", cnt_random_9)
    # print("cnt_random_10", cnt_random_10)

    return mutation_list, fcg



def check_feature(orignal_feature, new_feature):
    increase_index = []
    decrease_index = []
    orignal_zero = 0
    new_zero = 0
    for i in range(len(orignal_feature)):
        if orignal_feature[i] == 0:
            orignal_zero += 1

        if new_feature[i] == 0:
            new_zero += 1

        if orignal_feature[i] != new_feature[i]:
            # print(orignal_feature[i], new_feature[i])
            if orignal_feature[i] < new_feature[i]:
                increase_index.append(i)
            else:
                decrease_index.append(i)

    print("increase", len(increase_index))
    print("decrease", len(decrease_index))
    print("orignal_zero", orignal_zero)
    print("new_zero", new_zero)
    print("------------------")

    return increase_index, decrease_index

def delete_zero_feature(feature):
    new_feature = []
    for i in range(len(feature)):
        if feature[i] != 0:
            new_feature.append(feature[i])
    return new_feature

def one_mutation_m(fcg, steps, type):
    # csv_data = [
    #     ['degree', 'degree_increase_index', 'degree_decrease_index', 'katz', 'katz_increase_index',
    #      'katz_decrease_index', 'closeness', 'closeness_increase_index', 'closeness_decrease_index', 'harmonic',
    #      'harmonic_increase_index', 'closeness_decrease_index', 'MLP', 'benign_dis', 'malware_dis', 'benign_idx', 'malware_idx', 'diff', 'add_edge_label', 'intersection']
    # ]
    #
    # node_label = None
    #
    # print("senstive user node", fcg.sensitive_user_defined_nodes)
    # print("senstive system node", fcg.sensitive_system_nodes)

    # fcg.cal_centralities()
    # degree = fcg.degree_feature
    # if degree is None:
    #     print("degree_feature is none")
    #     return None
    # katz = fcg.katz_feature
    # closeness = fcg.closeness_feature
    # harmonic = fcg.harmonic_feature
    # combined_feature = np.concatenate((degree, katz, closeness, harmonic), axis=0)
    # combined_feature = combined_feature.reshape(1, -1)
    # benign_distance, malware_distance, benign_idx, malware_idx = get_distance_from_knn1(combined_feature)
    # print("benign_distance", benign_distance)
    # print("malware_distance", malware_distance)
    # print("benign_distance - malware_distance", benign_distance - malware_distance)
    #
    # pobs = test_model(combined_feature, 1, 'MLP', model_path='./430features_3yearsdataset_all/MLP.h5')
    # print("pobs", pobs)
    #
    #
    #
    # csv_data.append([delete_zero_feature(degree), None, None,
    #             delete_zero_feature(katz), None, None,
    #             delete_zero_feature(closeness), None, None,
    #             delete_zero_feature(harmonic), None, None,
    #             pobs[0][0], benign_distance, malware_distance, benign_idx, malware_idx, benign_distance - malware_distance, None, None])

    random_int = 0
    if type == 'add_edge':
        random_int = 1
    elif type == 'add_node':
        random_int = 2
    elif type == 'remove_node':
        random_int = 3
    elif type == 'rewiring':
        random_int = 4


    fcg = copy.deepcopy(fcg)
    #将给定的fcgs随机变换若干步,并保存
    # 获取一个 apk 的 call graph
    # print("fcg nodes", len(fcg.nodes))
    # print("fcg edges", len(fcg.edges))

    #每次传入是一个新的fcg

    # print("steps", steps)

    mutation_list = []

    i = 0
    error_cnt = 0
    while i < steps:
        #生成随机数1-4
        # print('iteration', i)
        # print("mutation ing edges", len(mutation.edges), len(fcg.edges))
        # print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))
        mutation = None
        intersection = None
        if random_int == 1:
            mutation = fcg.build_mutation_add_edge_type1()
        elif random_int == 2:
            mutation = fcg.build_mutation_add_node_type1()
        elif random_int == 3:
            mutation = fcg.build_mutation_remove_node_type1()
        elif random_int == 4:
            mutation = fcg.build_mutation_rewiring_type1(steps/100)

        # print(mutation.mutation)
        if mutation is None:
            # print("mutation failed")
            error_cnt += 1
            continue
        else:
            state = fcg.process_mutation(mutation)
            print("state", state)
            if state:
                mutation_list.append(mutation)

                # print("mutation", mutation.mutation)
                #
                # fcg.cal_centralities()
                # cur_degree = fcg.degree_feature
                # cur_katz = fcg.katz_feature
                # cur_closeness = fcg.closeness_feature
                # cur_harmonic = fcg.harmonic_feature
                # cur_combined_feature = np.concatenate((cur_degree, cur_katz, cur_closeness, cur_harmonic), axis=0)
                # cur_combined_feature = cur_combined_feature.reshape(1, -1)
                # cur_benign_distance, cur_malware_distance, cur_benign_idx, cur_malware_idx = get_distance_from_knn1(cur_combined_feature)
                # print("benign_distance", cur_benign_distance)
                # print("malware_distance", cur_malware_distance)
                # print("benign_distance - malware_distance", cur_benign_distance - cur_malware_distance)
                #
                # pobs = test_model(cur_combined_feature, 1, 'MLP', model_path='./430features_3yearsdataset_all/MLP.h5')
                # print("pobs", pobs)
                #
                # csv_data.append([delete_zero_feature(cur_degree), check_feature(degree, cur_degree)[0], check_feature(degree, cur_degree)[1],
                #                  delete_zero_feature(cur_katz), check_feature(katz, cur_katz)[0], check_feature(katz, cur_katz)[1],
                #                 delete_zero_feature(cur_closeness), check_feature(closeness, cur_closeness)[0], check_feature(closeness, cur_closeness)[1],
                #                 delete_zero_feature(cur_harmonic), check_feature(harmonic, cur_harmonic)[0], check_feature(harmonic, cur_harmonic)[1],
                #                 float(pobs[0][0]), float(cur_benign_distance), float(cur_malware_distance), cur_benign_idx, cur_malware_idx, float(cur_benign_distance - cur_malware_distance), node_label, intersection])
                #
                # print([delete_zero_feature(cur_degree), check_feature(degree, cur_degree)[0], check_feature(degree, cur_degree)[1],
                #                  delete_zero_feature(cur_katz), check_feature(katz, cur_katz)[0], check_feature(katz, cur_katz)[1],
                #                 delete_zero_feature(cur_closeness), check_feature(closeness, cur_closeness)[0], check_feature(closeness, cur_closeness)[1],
                #                 delete_zero_feature(cur_harmonic), check_feature(harmonic, cur_harmonic)[0], check_feature(harmonic, cur_harmonic)[1]])
                #
                # degree = cur_degree
                # katz = cur_katz
                # closeness = cur_closeness
                # harmonic = cur_harmonic

        if error_cnt > 10:
            print("all mutations failed")
            break


        i += 1
    # file_name = f'/data/a/shiwensong/dataset/analysis_one_success_process/{apk_name}_{feature_type}_experiment1_dec22.csv'
    # with open(file_name, 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerows(csv_data)

    return mutation_list, fcg

def ga(fcg, max_generation = 100, pop_num = 100, steps = 500, init_state = False, population = None, sensitive = True, sub = True, group = False):
    # inherit_prob: 个体直接成为下一代的概率，默认是0.3
    #population: pop_num个mutation_sequence

    save_state = False

    if not init_state:
        #1. init population
        population = _init_population(fcg, pop_num, steps, sensitive, group)#这个不会更改fcg的值
        pop_label_list, pop_score_list, mlp_pop_list, pop_shap_list = _population_score(fcg, population, False, sub)
        #sort the population， using score
        if init_num > pop_num:
            indexes = np.argsort(pop_score_list)[:pop_num]
            population = [population[i] for i in indexes]
            pop_score_list = [pop_score_list[i] for i in indexes]
            mlp_pop_list = [mlp_pop_list[i] for i in indexes]
            pop_shap_list = [pop_shap_list[i] for i in indexes]
            pop_label_list = [pop_label_list[i] for i in indexes]
            print("after init pop_score_list", len(pop_score_list))

        # print("pop_score_list", pop_score_list)
        # print("pop_shap_list", pop_shap_list)
        if len(pop_score_list) == 0:
            return None

    else:
        #从日志中恢复
        pop_label_list, pop_score_list, mlp_pop_list, pop_shap_list = _population_score(fcg, population, False, sub)
        print("pop_score_list", pop_score_list)
        print("pop_shap_list", pop_shap_list)
        if len(pop_score_list) == 0:
            return None

    #2. ga
    for i in range(max_generation):
        tf.keras.backend.clear_session()
        # 判断是否越狱
        #返回true/false
        generation_start = time.time()
        print("generation", i)

        #模拟测试
        # pop_score_list = read_log_for_every_individual(fcg)
        # print("pop_score_list", pop_score_list)
        # 修改两个个体


        #判断是否多轮的结果不变
        # state, score = check_population(fcg, pop_score_list)
        # print("state", state)
        # if state:
        #     one_index = pop_score_list.index(score)
        #     two_index = pop_score_list.index(score, one_index + 1)
        #     population[one_index].ga_mutation(fcg)
        #     population[two_index].ga_mutation(fcg)
        #     one_score = _individual_score(fcg, population[one_index])
        #     two_score = _individual_score(fcg, population[two_index])
        #     print("one_score", one_score)
        #     print("two_score", two_score)
        #     print("score", score)
        #     if (one_score != score or two_score != score) and (one_score is not None and two_score is not None):
        #         pop_score_list[one_index] = one_score
        #         pop_score_list[two_index] = two_score

        #2.1 judge if the target is found
        target_index = _fitness_function(pop_label_list)
        if target_index != -1:
            print("fitness function is true")
            save_log_for_every_individual(fcg, population[target_index], mlp_pop_list[target_index], i)
            break

        next_gene_population = []
        #note: for observe the change of every generation
        save_log_for_every_generation(fcg, mlp_pop_list, pop_shap_list, i, population)

        # print("every generation pop list", sorted(pop_score_list))

        # strategy 1: choose top 2,  strategy 2: former generation + new generation, choose top 100
        # top1, top2 = find_two_largest(pop_score_list)
        # top1, top2 = find_two_smallest(pop_score_list)
        # new_individual_top1 = copy.deepcopy(population[top1])
        # new_individual_top2 = copy.deepcopy(population[top2])
        # next_gene_population.append(new_individual_top1)
        # next_gene_population.append(new_individual_top2)
        # print("top1", pop_score_list[top1])
        # print("top2", pop_score_list[top2])

        #strategy 2: every 1 generation, update shap value



        #2.2 generate next generation
        while True:
            if len(next_gene_population) >= pop_num:
                next_gene_population = next_gene_population[:pop_num]
                break

            #2.3 random select two individual, using tornament selection
            #之前的参数是int(pop_num/2)-1
            individual1_index, individual2_index = _ga_select(population, pop_score_list, mlp_pop_list, pop_shap_list, int(pop_num/2)-1)
            individual1 = population[individual1_index]
            individual2 = population[individual2_index]
            individual1_score = pop_score_list[individual1_index]
            individual2_score = pop_score_list[individual2_index]
            #note: for observe the change of top 2
            print("individual1_score", individual1_score)
            print("individual2_score", individual2_score)

            #old strategy: 以这个individual1_score * inherit_prob概率，individual1直接成为下一代
            # if individual1_score * inherit_prob > random.random():
            #     new_individual = copy.deepcopy(individual1)
            #     next_gene_population.append(new_individual)
            #
            # if individual2_score * inherit_prob > random.random():
            #     new_individual = copy.deepcopy(individual2)
            #     next_gene_population.append(new_individual)

            #2.4 2 individuals crossover, and result is stored in individual1
            # new_individual1 is a list of mutation_sequence
            # print("before crossover!!!!")
            # print("individual1_group_list", len(individual1.final_group_list))
            # print("individual2_group_list", len(individual2.final_group_list))
            #create a new individual
            new_individual = copy.deepcopy(individual1)
            new_individual.crossover(individual2, fcg)
            # individual1.crossover_single(individual2, fcg)
            # print("new_individual_group_list", len(new_individual.final_group_list))
            # print("after crossover!!!!")
            #2.5 deal with conflict
            cur_fcg = deal_with_conflict(fcg, new_individual, group)# don't change fcg
            # print("after deal with conflict new_individual_group_list", len(new_individual.final_group_list))
            #2.6 ga mutation
            new_individual.ga_mutation(cur_fcg, 0.2, sensitive)
            # print("after mutation new_individual_group_list", len(new_individual.final_group_list))
            #2.7 update dependency
            if not group:
                new_individual.update_dependency()
            # print("after update dependency new_individual_group_list", len(new_individual.final_group_list))

            #2.8 append to next_gene_population
            next_gene_population.append(new_individual)

        # population = next_gene_population
        # choose top 100 from former generation and new generation
        # print("former score", sorted(mlp_pop_list))
        # print("former shap", sorted(mlp_pop_list))
        # print("")
        if i == max_generation - 1:
            save_state = True
        cur_pop_label_list, cur_pop_score_list, cur_mlp_pop_list, cur_pop_shap_list = _population_score(fcg, next_gene_population, save_state, sub)
        if len(cur_pop_score_list) == 0:#中止ga
            return None
        combined_population = population + next_gene_population
        combined_score_list = pop_score_list + cur_pop_score_list
        combined_shap_list = pop_shap_list + cur_pop_shap_list
        combined_mlp_list = mlp_pop_list + cur_mlp_pop_list
        combined_label_list = pop_label_list + cur_pop_label_list

        top_pop, top_score, top_mlp_score, top_shap, top_label = sort_all_pop(combined_population, combined_score_list, combined_mlp_list, combined_shap_list, combined_label_list, pop_num)
        population = top_pop
        pop_score_list = top_score
        pop_shap_list = top_shap
        mlp_pop_list = top_mlp_score
        pop_label_list = top_label

        # print("after score", sorted(pop_score_list))
        # print("after shap", sorted(pop_shap_list))
        # print("after mlp", sorted(mlp_pop_list))
        # print("len population", len(population))

        # generation_end = time.time()
        # with open('generation_time_with_group_init300_step100_jan23.txt', 'a') as file:
        #     file.write(str(i) + " " + str(generation_end - generation_start) + '\n')

    return population

def deal_with_conflict(fcg, individual1, control_group = False):
    # 处理完冲突之后，还处理了空的group

    #cannot change fcg value
    new_fcg = copy.deepcopy(fcg)
    # conflict_begin = time.time()
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
                # print("before solve conflict")
                # start = time.time()
                if not control_group:
                    state_conflict = individual1.solve_conflict(new_fcg, mutation, res)  # 同时修改了图
                    if not state_conflict:
                        # discard mutation
                        need_remove.append(i)
                        now_conflict += 1
                    ori_conflict += 1
                else:
                    need_remove.append(i)
                    ori_conflict += 1

        for idx in need_remove[::-1]:  # 倒序遍历索引
            del group[idx]

        if len(group) == 0:
            null_group_idx.append(j)

    for idx in null_group_idx[::-1]:  # 倒序遍历索引
        individual1.final_group_list.pop(idx)

    # conflict_end = time.time()
    # print("conflict time", conflict_end - conflict_begin)
    # print("original conflict cnt", ori_conflict)
    # print("now conflict cnt", now_conflict)
    # res = "original conflict cnt" + str(ori_conflict) + '\n' + "now conflict cnt" + str(now_conflict) + '\n'
    # with open('conflict_cnt_pop300_jan23.txt', 'a') as file:
    #     file.write(res)

    res = "withoutgroup conflict cnt" + str(ori_conflict) + '\n'
    with open('conflict_cnt_pop300_jan23.txt', 'a') as file:
        file.write(res)

    return new_fcg

def check_population(fcg, pop_score_list):
    print("check_population")
    current_score = sorted(pop_score_list)

    cnt = 0
    for i in range(len(current_score)):
        if current_score[i] == current_score[i-1]:
            cnt += 1
        if cnt > 5:
            score = current_score[i]
            return True, score



    file_path = fcg.apk_name + '_log_for_every_generation_430features_MLP.json'
    data = read_log_for_every_generation(file_path)
    if data is None:
        return False, None
    # print("data", len(data))

    score = None


    #获取最后一行
    if len(data) > 2:
        last_one_data = data[-2]
        last_one_score = last_one_data.get("score")
        last_one_score = sorted(last_one_score)
        print("last_one_score", last_one_score)

        last_two_data = data[-1]
        last_two_score = last_two_data.get("score")
        last_two_score = sorted(last_two_score)
        print("last_two_score", last_two_score)

        cnt = 0
        for i in range(len(current_score)):
            if current_score[i] == last_two_score[i] and last_two_score[i] == last_one_score[i]:
                cnt += 1
            if cnt > 10:
                score = current_score[i]
                return True, score
    return False, score


def find_two_largest(nums):
    if len(nums) < 2:
        raise ValueError("List must contain at least two elements.")

    max_index = second_max_index = -1
    max_val = second_max_val = float('-inf')

    for i, num in enumerate(nums):
        if num > max_val:
            second_max_val, second_max_index = max_val, max_index
            max_val, max_index = num, i
        elif num > second_max_val and num != max_val:
            second_max_val, second_max_index = num, i

    return max_index, second_max_index

def find_two_smallest(nums):
    if len(nums) < 2:
        raise ValueError("List must contain at least two elements.")

    min_index = second_min_index = -1
    min_val = second_min_val = float('inf')

    for i, num in enumerate(nums):
        if num < min_val:
            second_min_val, second_min_index = min_val, min_index
            min_val, min_index = num, i
        elif num < second_min_val and num != min_val:
            second_min_val, second_min_index = num, i

    return min_index, second_min_index

def discarding_conflict_after_crossover(fcg, final_group_list):
    #对final_group_list进行处理，把每个group里面的mutation都进行一次处理

    fcg = copy.deepcopy(fcg)

    error_cnt = 0
    for i in range(len(final_group_list)):
        group = final_group_list[i]
        # print("group", i)
        error_index = -1
        state = True

        for j in range(len(group)):
            # print("group[j]", len(group[j]))
            state = fcg.process_mutation(group[j])
            # print("state", state)
            # print("group[j]", group[j].mutation)
            if not state:
                error_index = j
                error_cnt += 1
                break

        # print("the group number", i)
        # for mutation in group:
        #     print(mutation.mutation)

        if not state:
            # 1.删除从error_index开始的group
            # print("error_index", error_index)
            # print("group len", len(group))
            print("error group len before", len(final_group_list[i]))
            final_group_list[i] = final_group_list[i][:error_index]
            print("error group len after", len(final_group_list[i]))
    print("error_cnt", error_cnt)

    for i in range(len(final_group_list) - 1, -1, -1):
        if len(final_group_list[i]) == 0:
            final_group_list.pop(i)

    return final_group_list

def check_conflict(fcg, mutation_list):
    print("check_conflict starting")
    fcg = copy.deepcopy(fcg)

    error_cnt = 0
    for i in range(len(mutation_list)):
        group = mutation_list[i]
        state = fcg.process_mutation_sequence(group)
        if not state:
            error_cnt += 1

    print("error_cnt", error_cnt)
    print("check_conflict ending")
    return error_cnt



def _individual_score(fcg, individual, save_state = False):
    fcg = copy.deepcopy(fcg)

    original_combined_feature = extract_feature_use_fcg(fcg)
    if original_combined_feature is None:
        print("degree_feature is none")
        return None
    # original_degree = fcg.degree_feature
    # if original_degree is None:
    #     print("degree_feature is none")
    #     return None
    # original_katz = fcg.katz_feature
    # original_closeness = fcg.closeness_feature
    # original_harmonic = fcg.harmonic_feature
    # original_combined_feature = np.hstack((original_degree, original_katz, original_closeness, original_harmonic))


    for j in range(len(individual.final_group_list)):
        group = individual.final_group_list[j]
        safe_mutation = []
        for i in range(len(group)):
            mutation = group[i]
            state, res = fcg.process_mutation(mutation)
            if state:
                safe_mutation.append(mutation)
            else:
                print("_individual_score mutation failed")
                # print(res)
                print("mutation", mutation.mutation)
        individual.final_group_list[j] = safe_mutation

    #1. cal 4 features
    combined_feature = extract_feature_use_fcg(fcg)
    if combined_feature is None:
        print("new degree_feature is none")
        return None

    print("original_combined_feature", original_combined_feature.shape)
    print("combined_feature", combined_feature.shape)

    return original_combined_feature, combined_feature, fcg

def calculate_opposite_adjustment(original_feature, now_feature, shap_value):
    #1. 所有的shap值
    # 调整方向与 SHAP 值相反
    # SHAP 值为正时，减小特征值（反向调整为负）
    # SHAP 值为负时，增加特征值（反向调整为正）
    # opposite_adjustment = -shap_value * (now_feature - original_feature)

    #2. top 10 shap值
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

    #取前10%的shap值
    # neg_index = neg_index[:int(len(neg_index)/10)]
    # pos_index = pos_index[:int(len(pos_index)/10)]

    #对应原来的index
    opposite_adjustment = 0.0
    for index in neg_index:
        opposite_adjustment = opposite_adjustment - negative_shap_value[index] * negative_diff_feature[index]

    for index in pos_index:
        opposite_adjustment = opposite_adjustment - positive_shap_value[index] * positive_diff_feature[index]


    return opposite_adjustment

def save_log_for_every_individual(fcg, individual, score, generation):
    # 提取每个 Mutation 对象的 mutation 字段
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


def save_graph_for_every_generation(fcg, state, generation):
    if state == 'failed':
        fcg.save(f'{save_dir}/ga_failed/', generation)
    elif state == 'success':
        fcg.save(f'{save_dir}/ga_success/', generation)

def save_log_for_every_generation(fcg, score, shap, generation, population):
    # 提取每个 Mutation 对象的 mutation 字段
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

def read_log_for_every_generation(file_name):
    # file_name = apk_name + '_log_for_every_generation.json'
    # print("file_name", file_name)


    if not os.path.exists(file_name):
        return None

    loaded_data = []

    # 从文件中逐行读取JSON字符串并反序列化为字典
    with open(file_name, 'r') as file:
        for line in file:
            data = json.loads(line.strip())  # 移除行尾的换行符并反序列化
            loaded_data.append(data)
    # print(loaded_data)
    return loaded_data

def read_log_for_first_generation(file_name):
    if not os.path.exists(file_name):
        return None

    with open(file_name, 'r') as file:
        for line in file:
            return json.loads(line.strip())
    return None

def read_log_for_last_generation(file_name):
    if not os.path.exists(file_name):
        return None

    last_line = None
    with open(file_name, 'r') as file:
        for line in file:
            last_line = line

    if last_line:
        return json.loads(last_line.strip())
    else:
        return None

def read_log_for_every_individual(fcg):
    file_name = fcg.apk_name + 'log_for_every_individual.json'
    if not os.path.exists(file_name):
        return None

    loaded_data = []

    # 从文件中逐行读取JSON字符串并反序列化为字典
    with open(file_name, 'r') as file:
        for line in file:
            data = json.loads(line.strip())  # 移除行尾的换行符并反序列化
            score = data.get('score')
            loaded_data.append(score)
    # print(loaded_data)
    return loaded_data[:100]

def extract_feature_use_fcg(fcg):
    fcg.cal_centralities(type)
    combined_feature = None
    if type == 'degree':
        combined_feature = fcg.degree_feature
    elif type == 'katz':
        combined_feature = fcg.katz_feature
    elif type == 'closeness':
        combined_feature = fcg.closeness_feature
    elif type == 'harmonic':
        combined_feature = fcg.harmonic_feature

    return np.array(combined_feature)

def _init_population(fcg, pop_num=100, steps = 500, sensitive = True, group = False):
    #随机生成pop_num个mutation_sequence
    population = []
    shap_sum_list = []

    #初始化种群
    # 1. 加100个点，每个点500步
    # 2. rewiring延长到sensitive的路径，每个边500步

    print("init population")

    try_times = 0
    while try_times < init_num:
        print("individual: ", try_times)
        #每次调用random_m，都是一个新的fcg, 返回一个individual里面存储500个mutation_sequence，mutation对象用来创建一个mutationSequence
        mutation_lists, cur_fcg = random_m(fcg, steps, sensitive)

        # print("original len", len(mutation_lists))
        mutationSequence = MutationSequence(mutation_lists, group)
        # mutationSequence.generate_group_list()
        # print("len group list", len(mutationSequence.final_group_list))
        #把group_list加入到population里面
        population.append(mutationSequence)
        try_times += 1

    #select top 100
    print("population len", len(population))
    print("finished init population")

    return population

def _fitness_function(pop_score_list):
    target_index = -1
    # print("pop_score_list", pop_score_list)
    for i in range(len(pop_score_list)):
        #一个score代表是malware的概率，越小越好，小于0说明是benign
        if pop_score_list[i] is not None and pop_score_list[i] == 0:
            target_index = i
            break
    return target_index

def _population_score(fcg, pop, save_state = False, sub = True):
    # version1: shap+score
    score_list = []
    shap_list = []
    MLP_score_list = []
    pop_feature_list = []
    pop_fcg_list = []
    y_label_list = []
    # 使用 range() 函数以倒序索引列表
    for i in range(len(pop) - 1, -1, -1):
        print("loop individual for cal score: ", i)

        res = _individual_score(fcg, pop[i], save_state)
        # if score is None or shap_sum is None:
        if res is None:
            # 如果 score 是 None，则删除 pop 中的对应元素
            del pop[i]
        else:
            original_feature, combined_feature, cur_fcg = res
            # 否则，将 score 添加到 score_list
            shap_sum = calculate_opposite_adjustment(original_feature, combined_feature, fcg.shap_value)
            combined_feature = combined_feature.reshape(1, -1)
            pop_feature_list.extend(combined_feature)
            pop_fcg_list.append(cur_fcg)

            shap_list.append(shap_sum)

    pop_feature_list = np.array(pop_feature_list)
    print("pop_feature_list", pop_feature_list.shape)
    if sub:
        all_mlp_scores = MLP_model.predict(pop_feature_list)
    else:
        all_mlp_scores = len(pop) * [[0]]
    all_benign_distance, all_benign_indices = benign_knn_model.kneighbors(pop_feature_list)
    all_malware_distance, all_malware_indices = malware_knn_model.kneighbors(pop_feature_list)
    all_Y_pred = model.predict(pop_feature_list)

    for i in range(len(pop)):
        # top2 benign distance
        benign_distance = ( all_benign_distance[i][0] + all_benign_distance[i][1] ) / 2.0
        # top2 malware distance
        malware_distance = ( all_malware_distance[i][0] + all_malware_distance[i][1] ) / 2.0
        # distance, 越小越好
        score = benign_distance - malware_distance
        score_list.append(score)
        # MLP score
        MLP_score_list.append(all_mlp_scores[i][0])

        cur_fcg = pop_fcg_list[i]

        #y_label
        cur_label = all_Y_pred[i]
        y_label_list.append(cur_label)

        if cur_label == 0:
            save_graph_for_every_generation(cur_fcg, 'success', -1)

        if save_state:
            save_graph_for_every_generation(cur_fcg, 'failed', 39)

    # print("score_list", score_list)
    # print("MLP_score_list", MLP_score_list)
    # print("shap_list", shap_list)

    # 返回修改后的 score_list
    return y_label_list[::-1], score_list[::-1], MLP_score_list[::-1], shap_list[::-1]

    #version2: update shap value
    # start = time.time()
    # shap_list = []
    # score_list = []
    # original_features = []
    # new_features = []
    # for i in range(len(pop) - 1, -1, -1):
    #     print("loop individual for cal score: ", i)
    #     individual = pop[i]
    #     individual_fcg = copy.deepcopy(fcg)
    #     #1. cal original 4 features
    #     individual_fcg.cal_centralities()
    #     original_degree = individual_fcg.degree_feature
    #     if original_degree is not None:
    #         original_katz = individual_fcg.katz_feature
    #         original_closeness = individual_fcg.closeness_feature
    #         original_harmonic = individual_fcg.harmonic_feature
    #         original_combined_feature = np.hstack((original_degree, original_katz, original_closeness, original_harmonic))
    #
    #
    #         #2. cal now 4 features, may exist conflict
    #         for j in range(len(individual.final_group_list)):
    #             group = individual.final_group_list[j]
    #             safe_mutation = []
    #             for k in range(len(group)):
    #                 mutation = group[k]
    #                 state, res = individual_fcg.process_mutation(mutation)
    #                 if state:
    #                     safe_mutation.append(mutation)
    #             individual.final_group_list[j] = safe_mutation
    #
    #         state = individual_fcg.cal_centralities()
    #         if state and individual_fcg.degree_feature is not None:
    #             degree_feature = individual_fcg.degree_feature
    #             katz_feature = individual_fcg.katz_feature
    #             closeness_feature = individual_fcg.closeness_feature
    #             harmonic_feature = individual_fcg.harmonic_feature
    #             combined_feature = np.hstack((degree_feature, katz_feature, closeness_feature, harmonic_feature))
    #
    #             original_features.append(original_combined_feature)
    #             new_features.append(combined_feature)
    #
    #         else:
    #             del pop[i]
    #
    #     else:
    #         del pop[i]
    #
    #     #cal shap value
    #     if len(original_features) != 0 and len(new_features) != 0:
    #         new_features = np.array(new_features)
    #         shap_values = get_shap(train_dataset, new_features, MLP_model)
    #         shap_values = shap_values[0]
    #         # print("shap_values population score", shap_values.shape)
    #
    #         #cal shap sum
    #         for i in range(len(original_features)):
    #             shap_sum = calculate_opposite_adjustment(original_features[i], new_features[i], shap_values[i])
    #             shap_list.append(shap_sum)
    #             score = test_model(new_features[i].reshape(1, -1), 1, 'MLP', model_path='./430features_3yearsdataset_all/MLP.h5')
    #             print("score", score)
    #             score_list.append(score[0][0])
    #
    #
    #     end = time.time()
    #     print("cal shap time", end - start)
    #     # 返回修改后的 score_list
    #     return score_list, shap_list

# def _ga_select(pop, pop_score_list):
#     # 计算总分
#     total_score = sum(pop_score_list)
#
#     # 计算每个个体的选择概率
#     selection_probabilities = [score / total_score for score in pop_score_list]
#
#     # 选择第一个个体的索引
#     individual1_index = _roulette_select(selection_probabilities)
#
#     # 将第一个选择的个体的概率设置为0，并重新计算概率
#     selection_probabilities[individual1_index] = 0
#     total_score -= pop_score_list[individual1_index]
#     selection_probabilities = [score / total_score if total_score > 0 else 0 for score in pop_score_list]
#
#     # 选择第二个个体的索引
#     individual2_index = _roulette_select(selection_probabilities)
#
#     return individual1_index, individual2_index

def select(pop_score_list, pop_mlp_list, pop_shap_list, idxes=None):

    # dominate version
    pop_mlp_candidate = [pop_mlp_list[idx] for idx in idxes]
    pop_score_candidate = [pop_score_list[idx] for idx in idxes]
    pop_shap_candidate = [pop_shap_list[idx] for idx in idxes]
    # print("pop_score_candidate", pop_score_candidate)
    # print("pop_shap_candidate", pop_shap_candidate)
    if len(idxes) == 1:
        return idxes[0]

    # 把score弄成反向的分数，越小越好
    com = []
    for i in range(len(pop_score_candidate)):
        score = pop_score_candidate[i]
        mlp = pop_mlp_candidate[i]
        shap = -pop_shap_candidate[i]
        com.append([score, shap, mlp])

    # print("com", com)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=com)

    return idxes[ndf[0][0]]

    # sum version
    # pop_score_candidate = [pop_score_list[idx] for idx in idxes]
    # pop_shap_candidate = [pop_shap_list[idx] for idx in idxes]
    # # print("pop_score_candidate", pop_score_candidate)
    # # print("pop_shap_candidate", pop_shap_candidate)
    # if len(idxes) == 1:
    #     return idxes[0]
    #
    # # 把shap弄成反向的分数，越小越好
    # com = []
    # for i in range(len(pop_score_candidate)):
    #     score = pop_score_candidate[i]
    #     shap = -pop_shap_candidate[i]
    #     com.append(score + shap)
    #
    # # print("com", com)
    # min_idx = np.argmin(com)
    #
    # return idxes[min_idx]

    # only shap version
    # pop_shap_candidate = [pop_shap_list[idx] for idx in idxes]
    # # print("pop_score_candidate", pop_score_candidate)
    # # print("pop_shap_candidate", pop_shap_candidate)
    # if len(idxes) == 1:
    #     return idxes[0]
    #
    # max_shap_idx = np.argmax(pop_shap_candidate)
    #
    # return idxes[max_shap_idx]

    #only score version
    # pop_score_candidate = [pop_score_list[idx] for idx in idxes]
    # # print("pop_score_candidate", pop_score_candidate)
    # # print("pop_shap_candidate", pop_shap_candidate)
    # if len(idxes) == 1:
    #     return idxes[0]
    #
    # min_score_idx = np.argmin(pop_score_candidate)
    # # print("idxes[min_score_idx]", idxes[min_score_idx])
    #
    # return idxes[min_score_idx]

def sort_all_pop(pop, pop_score_list, pop_mlp_list, pop_shap_list, pop_label_list, pop_num):
    # dominate
    # 把shap弄成反向的分数，越小越好
    #dominate version
    com = []
    for i in range(len(pop_score_list)):
        mlp = pop_mlp_list[i]
        score = pop_score_list[i]
        shap = -pop_shap_list[i]
        com.append([score, mlp, shap])

    # print("com", com)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=com)
    # print("ndf", ndf)
    pop_total = []
    pop_score_list_total = []
    pop_shap_list_total = []
    pop_mlp_list_total = []
    pop_label_list_total = []
    for level in ndf:
        for idx in level:
            # print("idx", idx)
            pop_total.append(pop[idx])
            pop_score_list_total.append(pop_score_list[idx])
            pop_shap_list_total.append(pop_shap_list[idx])
            pop_mlp_list_total.append(pop_mlp_list[idx])
            pop_label_list_total.append(pop_label_list[idx])
            if len(pop_total) >= pop_num:
                break

        if len(pop_total) >= pop_num:
            break

    return pop_total, pop_score_list_total, pop_mlp_list_total, pop_shap_list_total, pop_label_list_total

    # com = []
    # for i in range(len(pop_score_list)):
    #     score = pop_score_list[i]
    #     shap = -pop_shap_list[i]
    #     com.append(score + shap)
    #
    # # print("com", com)
    # indexes = np.argsort(com)
    # # print("ndf", ndf)
    # pop_total = []
    # pop_score_list_total = []
    # pop_shap_list_total = []
    # for idx in indexes:
    #     # print("idx", idx)
    #     pop_total.append(pop[idx])
    #     pop_score_list_total.append(pop_score_list[idx])
    #     pop_shap_list_total.append(pop_shap_list[idx])
    #     if len(pop_total) >= pop_num:
    #         break
    #
    # return pop_total, pop_score_list_total, pop_shap_list_total

    #only shap version
    #from large to small
    # indexes = np.argsort(pop_shap_list)[::-1]
    # # print("ndf", ndf)
    # pop_total = []
    # pop_score_list_total = []
    # pop_shap_list_total = []
    # for idx in indexes:
    #     # print("idx", idx)
    #     pop_total.append(pop[idx])
    #     pop_score_list_total.append(pop_score_list[idx])
    #     pop_shap_list_total.append(pop_shap_list[idx])
    #     if len(pop_total) >= pop_num:
    #         break
    #
    # return pop_total, pop_score_list_total, pop_shap_list_total

    #only score version
    # indexes = np.argsort(pop_mlp_list)
    # # print("ndf", ndf)
    # pop_total = []
    # pop_score_list_total = []
    # pop_shap_list_total = []
    # pop_mlp_list_total = []
    # for idx in indexes:
    #     # print("idx", idx)
    #     pop_total.append(pop[idx])
    #     pop_score_list_total.append(pop_score_list[idx])
    #     pop_shap_list_total.append(pop_shap_list[idx])
    #     pop_mlp_list_total.append(pop_mlp_list[idx])
    #     if len(pop_total) >= pop_num:
    #         break
    #
    # return pop_total, pop_score_list_total, pop_mlp_list_total, pop_shap_list_total

def _ga_select(pop, pop_score_list, pop_mlp_list, pop_shap_list, tournament_size=20):#的分越小，概率越大
    # 确保锦标赛大小不超过种群大小
    tournament_size = min(tournament_size, len(pop))
    print(pop_score_list)
    print(pop_shap_list)

    # 选择两个个体的下标
    winner_indices = set()
    while True:
        # 随机选择tournament_size个个体
        # tournament = random.sample(list(enumerate(pop_score_list)), tournament_size)
        tournament_indices = random.sample(range(len(pop)), tournament_size)
        # 选出具有最低评分的个体的下标
        # winner_index = min(tournament, key=lambda x: x[1])[0]
        # winner_index = min(tournament_indices, key=lambda idx: composite_score(pop_score_list[idx], pop_shap_list[idx]))
        # 综合考虑预测概率和SHAP值一致性
        winner_index = select(pop_score_list, pop_mlp_list, pop_shap_list, tournament_indices)
        # 添加到获胜者下标列表中
        winner_indices.add(winner_index)
        # print("winner_index", winner_index)
        # print("tournament_indices", tournament_indices)
        # print("pop_score_list", pop_score_list)
        # print("pop_shap_list", pop_shap_list)
        if len(winner_indices) >= 2:
            break

    winner_indices = list(winner_indices)

    return winner_indices[0], winner_indices[1]

def rank(x):
    y = [0]*len(x)
    ranks = np.argsort(x)
    for i in range(len(x)):
        y[ranks[i]]=i
    return y

def _ga_select_old(pop, pop_score_list):#得分越小，概率越大
    # 获取最大得分以便从中减去
    max_score = max(pop_score_list)

    # 计算每个个体的选择概率（得分越低，概率越高）
    selection_probabilities = [(max_score - score) for score in pop_score_list]
    total_score = sum(selection_probabilities)

    # 防止总分为0的情况
    if total_score == 0:
        return random.randint(0, len(pop) - 1), random.randint(0, len(pop) - 1)

    # 计算每个个体的选择概率
    selection_probabilities = [score / total_score for score in selection_probabilities]

    # 选择第一个个体的索引
    individual1_index = _roulette_select(selection_probabilities)

    # 将第一个选择的个体的概率设置为0，并重新计算概率
    total_score -= selection_probabilities[individual1_index] * total_score
    selection_probabilities[individual1_index] = 0
    selection_probabilities = [score / total_score if total_score > 0 else 0 for score in selection_probabilities]

    # 选择第二个个体的索引
    individual2_index = _roulette_select(selection_probabilities)

    return individual1_index, individual2_index


def _roulette_select(probabilities):
    # 累积求和
    cumulative_sum = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]
    # 随机选择一个数
    r = random.random()

    # 根据轮盘赌选择个体的索引
    for i, s in enumerate(cumulative_sum):
        if r <= s:
            return i

    # 添加的返回语句
    return len(probabilities) - 1

def custom_softmax_with_knn(current_sample):
    """
    使用KNN检测器获取当前样本的k个最近邻居，并计算类别0和类别1的softmax概率。

    :param knn_detector: KNN检测器实例
    :param current_sample: 当前样本
    :param k: 最近邻居的数量，默认为5
    :return: 类别0和类别1的概率
    """
    # 使用KNN检测器获取最近的5个邻居的标签
    # 1.所有的begin，knn1,top1-benign
    # 2.所有的malware,knn1,top1-malware
    #sample knn1

    # best_model = joblib.load(model_path)
    #
    #
    #
    # nearest_neighbors = knn_detector.get_nearest_neighbors(current_sample, k)
    # nearest_labels = [neighbor.label for neighbor in nearest_neighbors]
    #
    # # 计算类别0和类别1的数量
    # counts = [nearest_labels.count(0), nearest_labels.count(1)]
    #
    # # 应用softmax
    # probabilities = softmax(counts)
    #
    # return probabilities[0], probabilities[1]
    pass

def print_fcg_information(fcg):
    print("fcg nodes", len(fcg.nodes))
    print("fcg edges", len(fcg.edges))
    print("fcg boundary edges", len(fcg.boundary_edges))
    print("fcg system nodes", len(fcg.system_nodes))
    print("fcg apk name", fcg.apk_name)

def deal_with_shap(fcg, shap_value):
    degree_shap = shap_value[:430]
    katz_shap = shap_value[430:860]
    closeness_shap = shap_value[860:1290]
    harmonic_shap = shap_value[1290:1720]

    ori_degree = fcg.degree_feature
    ori_katz = fcg.katz_feature
    ori_closeness = fcg.closeness_feature
    ori_harmonic = fcg.harmonic_feature

    shap_sum = 0.0

    res_add_node = []
    # for i in range(430):
    #     shap_sum = shap_sum + degree_shap[i]*ori_degree[i] + katz_shap[i]*ori_katz[i] + closeness_shap[i]*ori_closeness[i] + harmonic_shap[i]*ori_harmonic[i]
    #     if degree_shap[i] != 0 or katz_shap[i] != 0 or closeness_shap[i] != 0 or harmonic_shap[i] != 0:
    #         print("node", i)
    #         print("degree_shap", degree_shap[i])
    #         print("katz_shap", katz_shap[i])
    #         print("closeness_shap", closeness_shap[i])
    #         print("harmonic_shap", harmonic_shap[i])
        # for j in range(1):
        #     print("j", j)
        #     mutation = fcg.build_mutation_add_node()
        #     fcg.process_mutation(mutation)
        #     fcg.cal_centralities()
        #     degree = fcg.degree_feature
        #     katz = fcg.katz_feature
        #     closeness = fcg.closeness_feature
        #     harmonic = fcg.harmonic_feature
        #     combined_feature = np.hstack((degree, katz, closeness, harmonic))
        #     combined_feature = combined_feature.reshape(1, -1)
        #     Y_probs = test_model(combined_feature, 1, 'MLP',
        #                          model_path='./430features_3yearsdataset_all/MLP.h5')
        #     print("pos Y_probs", Y_probs[0][0])
        #     res_add_node.append(Y_probs[0][0])


        # if degree_shap[i] <= 0 and katz_shap[i] <= 0 and closeness_shap[i] <= 0 and harmonic_shap[i] <= 0:
        #     #对当前点 add edge score一直上升，当shap值全都是负向或者全都是正向
        #     print("i", i)
        #     end_node = fcg.sensitive_nodes[i]
        #     if end_node == -1:
        #         print("end_node is -1")
        #         continue
        #
        #     callers = [m for m, n in fcg.edges if n == end_node and m != n]
        #     new_node = 20000
        #     fcg.nodes.add(new_node)
        #     fcg.user_defined_nodes.add(new_node)
        #     fcg.edges.add((new_node, end_node))
        #     fcg.cal_centralities()
        #     degree = fcg.degree_feature
        #     katz = fcg.katz_feature
        #     closeness = fcg.closeness_feature
        #     harmonic = fcg.harmonic_feature
        #     combined_feature = np.hstack((degree, katz, closeness, harmonic))
        #     combined_feature = combined_feature.reshape(1, -1)
        #     Y_probs = test_model(combined_feature, 1, 'MLP',
        #                             model_path='./430features_3yearsdataset_all/MLP.h5')
        #     print("add node Y_probs", Y_probs[0][0])

            # for begin_node in callers:
            #     fcg.current_call_graph.remove_edge(begin_node, end_node)
            #     fcg.edges.remove((begin_node, end_node))
            #     fcg.edges.add((begin_node, new_node))
            #     fcg.cal_centralities()
            #     degree = fcg.degree_feature
            #     katz = fcg.katz_feature
            #     closeness = fcg.closeness_feature
            #     harmonic = fcg.harmonic_feature
            #     combined_feature = np.hstack((degree, katz, closeness, harmonic))
            #     combined_feature = combined_feature.reshape(1, -1)
            #     Y_probs = test_model(combined_feature, 1, 'MLP',
            #                          model_path='./430features_3yearsdataset_all/MLP.h5')
            #     print("neg Y_probs", Y_probs[0][0])
            #     #分析与原来特征的关系
            #     shap_sum = -(degree_shap[i]*(degree[i] - ori_degree[i]) + katz_shap[i]*(katz[i] - ori_katz[i]) + closeness_shap[i]*(closeness[i] - ori_closeness[i]) + harmonic_shap[i]*(harmonic[i] - ori_harmonic[i]))
            #     print("shap_sum", shap_sum)
            #     ori_degree = degree
            #     ori_katz = katz
            #     ori_closeness = closeness
            #     ori_harmonic = harmonic

    # print("shap_sum", shap_sum)
    # # 将列表存储为 .pkl 文件
    # with open('res_add_node.pkl', 'wb') as file:
    #     pickle.dump(res_add_node, file)


        # for j in range(100):
        #     print("j", j)
        #     mutation = fcg.build_mutation_add_node()
        #     fcg.process_mutation(mutation)
        #     fcg.cal_centralities()
        #     degree = fcg.degree_feature
        #     katz = fcg.katz_feature
        #     closeness = fcg.closeness_feature
        #     harmonic = fcg.harmonic_feature
        #     combined_feature = np.hstack((degree, katz, closeness, harmonic))
        #     combined_feature = combined_feature.reshape(1, -1)
        #     Y_probs = test_model(combined_feature, 1, 'MLP',
        #                          model_path='./430features_3yearsdataset_all/MLP.h5')
        #     print("add node Y_probs", Y_probs[0][0])

            #20000

        # callers = [m for m, n in fcg.edges if n == end_node and m != n]
        # print("callers", len(callers))
        # if len(callers) < 2:
        #     continue
        # #
        # # number = 0
        # for j in range(len(callers) - 1, 0, -1):
        #     begin_node = callers[j]
        #     former_begin_node = callers[j-1]
        #     if begin_node in fcg.system_nodes or former_begin_node in fcg.system_nodes:
        #         continue
        #
        #     #延长到sensitive的路径
        #     # mid_node = number + 20000
        #     # number += 1
        #     # if begin_node != mid_node and end_node != mid_node:
        #     #     fcg.edges.remove((begin_node, end_node))
        #     #     fcg.edges.add((begin_node, mid_node))
        #     #     fcg.edges.add((mid_node, end_node))
        #
        #     #merge减少被调用的次数
        #
            # caller_benign_node = [m for m, n in fcg.edges if n == begin_node and m != n]
            # callee_benign_node = [n for m, n in fcg.edges if m == begin_node and m != n]
            # print("caller_benign_node", len(caller_benign_node))
            # if len(caller_benign_node) == 0:
            #     continue
            #
            # for caller in caller_benign_node:
            #     if caller not in fcg.system_nodes:
            #         fcg.edges.remove((caller, begin_node))
            #         fcg.edges.add((caller, former_begin_node))
            #
            # for callee in callee_benign_node:
            #     fcg.edges.remove((begin_node, callee))
            #     fcg.edges.add((former_begin_node, callee))
            #
            # if (begin_node, end_node) in fcg.edges:
            #     fcg.edges.remove((begin_node, end_node))
            # if begin_node in fcg.nodes:
            #     fcg.nodes.remove(begin_node)
            # if begin_node in fcg.user_defined_nodes:
            #     fcg.user_defined_nodes.remove(begin_node)
            #
            # fcg.cal_centralities()
            # degree = fcg.degree_feature
            # katz = fcg.katz_feature
            # closeness = fcg.closeness_feature
            # harmonic = fcg.harmonic_feature
            # combined_feature = np.hstack((degree, katz, closeness, harmonic))
            # combined_feature = combined_feature.reshape(1, -1)
            # Y_probs = test_model(combined_feature, 1, 'MLP',
            #                      model_path='./430features_3yearsdataset_all/MLP.h5')
            # print("merge Y_probs", Y_probs[0][0])

        # elif degree_shap[i] <= 0 and katz_shap[i] <= 0 and closeness_shap[i] <= 0 and harmonic_shap[i] <= 0:

def load_MLP_model(model_path):
    # 定义新模型结构，与原始模型相同，但最后一个 Dense 层没有激活函数
    MLP2 = Sequential([
        Dense(128, activation='relu', input_shape=(430,)),  # 同样的输入维度
        Dense(64, activation='relu'),
        Dense(1)  # 没有激活函数
    ])

    # 加载原始模型
    original_model = tf.keras.models.load_model(model_path)

    # 将原始模型的权重复制到新模型的相应层
    # 这里我们假设层的数量和顺序是完全匹配的
    for layer, original_layer in zip(MLP2.layers, original_model.layers):
        layer.set_weights(original_layer.get_weights())

    # 现在，MLP2 拥有了 MLP 的权重，但最后一层没有激活函数

    return MLP2

def test_MLP_model(vector):
    #model is a global variable
    Y_pred_probs = model.predict(vector)  # 预测概率
    return Y_pred_probs


def obtain_dataset_for_shap():
    # obtain sample shap value
    train_feature_2018 = obtain_dataset('2018')
    train_feature_2019 = obtain_dataset('2022')
    train_feature_2020 = obtain_dataset('2023')
    train_feature_2018_df = pd.DataFrame(train_feature_2018)
    train_feature_2019_df = pd.DataFrame(train_feature_2019)
    train_feature_2020_df = pd.DataFrame(train_feature_2020)

    train_dataset = pd.concat([train_feature_2018_df, train_feature_2019_df, train_feature_2020_df], axis=0)
    print("train_dataset", train_dataset.shape)
    return train_dataset

def recover_graph(original_graph_path, shap_value, failed_graph_dir):
    sample_name = original_graph_path.split("/")[-1].split(".")[0]
    print("sample_name", sample_name)
    # 1. load failed graphs
    failed_graphs = glob.glob(failed_graph_dir + sample_name + '*.gexf')
    print(len(failed_graphs))

    if len(failed_graphs) == 0:
        print("no failed graph")
        return None

    # 2. load original graph
    original_fcg = FCG(original_graph_path, 1, shap_value)

    # 3. cal the scores, and find the best 1 failed graph
    min_score = 100000
    min_failed_graph = None
    failed_graph_scores = []
    for failed_path in failed_graphs:
        nodes, edges = load_graph(failed_path)

        current_fcg = copy.deepcopy(original_fcg)
        current_fcg.nodes = nodes
        current_fcg.edges = edges

        current_fcg.cal_centralities(type)
        combined_feature = None
        if type == 'degree':
            combined_feature = current_fcg.degree_feature
        elif type == 'katz':
            combined_feature = current_fcg.katz_feature
        elif type == 'closeness':
            combined_feature = current_fcg.closeness_feature
        elif type == 'harmonic':
            combined_feature = current_fcg.harmonic_feature

        combined_feature = np.array(combined_feature)
        combined_feature = combined_feature.reshape(1, -1)

        all_benign_distance, all_benign_indices = benign_knn_model.kneighbors(combined_feature)
        all_malware_distance, all_malware_indices = malware_knn_model.kneighbors(combined_feature)
        benign_distance = (all_benign_distance[0][0] + all_benign_distance[0][1]) / 2.0
        malware_distance = (all_malware_distance[0][0] + all_malware_distance[0][1]) / 2.0
        score = benign_distance - malware_distance
        print("Y_probs",score)

        # min_failed_graph = deal_with_graph(min_failed_graph)


        if score < min_score:
            min_score = score
            min_failed_graph = current_fcg
        # failed_graph_scores.append((Y_probs[0][0], current_fcg))
        # break

    # top_failed_graphs = sorted(failed_graph_scores, key=lambda x: x[0])
    # print("top_failed_graphs", top_failed_graphs)

    # for dealing with the nodes
    # min_failed_graph = top_failed_graphs[k][1]
    min_failed_graph = deal_with_graph(min_failed_graph)
    return min_failed_graph

def deal_with_graph(min_failed_graph):
    try:

        for node in min_failed_graph.nodes:
            if node not in min_failed_graph.user_defined_nodes and node not in min_failed_graph.system_nodes:
                min_failed_graph.user_defined_nodes.add(node)

        remove_nodes = []
        for node in min_failed_graph.user_defined_nodes:
            if node not in min_failed_graph.nodes:
                remove_nodes.append(node)

        for node in remove_nodes:
            min_failed_graph.user_defined_nodes.remove(node)


        min_failed_graph.init_sensitive_upstream()

        return min_failed_graph
    except:
        return None

def load_graph(file_path):
    # 假设 G_loaded 是从 GEXF 文件中加载的图
    # 从GEXF文件中加载图
    G_loaded = nx.read_gexf(file_path)

    # 创建两个列表用于存储整数类型的节点和边
    nodes = set()
    edges = set()

    # 遍历加载的图中的所有节点，并将节点标识符转换为整数后添加到nodes列表中
    for node in G_loaded.nodes():
        nodes.add(int(node))

    # 遍历加载的图中的所有边，并将边的节点标识符转换为整数后添加到edges列表中
    for edge in G_loaded.edges():
        edges.add((int(edge[0]), int(edge[1])))

    return nodes, edges

def check_is_zero(vector):
    cnt = 0
    for i in range(len(vector)):
        if vector[i] != 0:
            cnt += 1
    return cnt
    
def check_sampe(names, slices):
    for i in slices:
        name = names[i]
        print(name+" ", end="")
        
def load_benign_and_malware_knn_model():
    benign_path = f'./430features_6yearsdataset_1000samples/{type}_benignknn_10.pkl'
    malware_path = f'./430features_6yearsdataset_1000samples/{type}_malwareknn_10.pkl'

    benign_knn_model = joblib.load(benign_path)
    malware_knn_model = joblib.load(malware_path)
    return benign_knn_model, malware_knn_model

def obtain_all_failed_sample_name():
    individual_file_names = glob.glob(recover_dir + "/*for_every_individual.json")
    generation_file_names = glob.glob(recover_dir + "/*for_every_generation.json")
    print("individual_file_names", len(individual_file_names))
    print("population_file_names", len(generation_file_names))
    individual = set()
    generation = set()
    for name in individual_file_names:
        file_name = name.replace("_log_for_every_individual.json", "")
        file_name = file_name.split("/")[-1]
        # print(file_name)
        individual.add(file_name)
    for name in generation_file_names:
        file_name = name.replace("_log_for_every_generation.json", "")
        file_name = file_name.split("/")[-1]
        generation.add(file_name)
    # print("individual", len(individual))
    # print("generation", len(generation))
    # print("individual", individual)
    intersection = individual.intersection(generation)
    print("intersection", len(intersection))
    union = individual.union(generation)
    print("union", len(union))
    failed = generation - intersection
    print("failed", len(failed))
    return failed, union

if __name__ == '__main__':
    # 获取当前进程
    p = psutil.Process(os.getpid())

    # 设置进程的亲和性，这里是使用 CPU 1 到 16
    p.cpu_affinity(list(range(1, 110)))

    # 获取传入参数，proc_index
    if len(sys.argv) > 2:
        # 尝试将第一个参数转换为整数
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

    # read attack samples from txt file
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    name = []
    for year in years:
        txt_file = f'{feature_dir}/mar10_attack_samples_{year}_60.txt'
        with open(txt_file, 'r') as file:
            tmp = file.readlines()
        tmp = tmp[:20]
        name.extend(tmp)

    print("selected_samples", len(name))

    # obtain all sample name
    all_sample_name, union = obtain_all_failed_sample_name()

    failed_sample = []
    for i in range(len(name)):
        # obtain apk_name
        apk_name = name[i].split('/')[-1].split('.gexf')[0]
        if apk_name in all_sample_name and apk_name != '01411844d63ecc8ef3a03069d314ee6f3d4064e7a40cfa9afa90ea74e3e96182':
            failed_sample.append(name[i])

    name = failed_sample
    print("failed_sample", len(name))

    # ignore_sample = []
    # for i in range(len(name)):
    #     apk_name = name[i].split('/')[-1].split('.gexf')[0]
    #     if apk_name not in union:
    #         ignore_sample.append(name[i])
    # name = ignore_sample
    # print("ignore_sample", len(name))

    part_size = len(name) // proc_total
    # 计算需要额外分配一个元素的部分的数量
    extra = len(name) % proc_total
    # 计算当前进程的起始索引
    start_index = proc_id * part_size + min(proc_id, extra)
    # 如果proc_id小于extra，意味着这个部分需要额外增加一个元素
    if proc_id < extra:
        part_size += 1
    # 计算结束索引
    end_index = start_index + part_size

    # for cal shap value
    MLP_model = load_MLP_model(f'./430features_6yearsdataset_1000samples/{type}MLP_{model_name}.h5')
    MLP_model2 = load_model(f'./430features_6yearsdataset_1000samples/{type}MLP_{model_name}.h5')
    # print("MLP_model", MLP_model.summary())

    train_dataset_dir = f'select_6years_train_dataset_for_shap_{type}.csv'
    train_dataset = pd.read_csv(train_dataset_dir)
    print("train_dataset", train_dataset.shape)
    # 随机选择100个
    train_dataset = train_dataset.sample(n=100, random_state=42)
    # #
    # # # load model, global variable
    model = load_model(f'./430features_6yearsdataset_1000samples/{type}{model_name}.pkl')

    # generate the new shap for the failed samples
    all_samples_features = []

    zeros_list = [0] * 430

    benign_knn_model, malware_knn_model = load_benign_and_malware_knn_model()


    # #load shap
    shap_dir = f'{save_dir}120_samples_{model_name}_{type}.pkl'
    with open(shap_dir, 'rb') as file:
        new_shap_values = pickle.load(file)
    print("new_shap_values", new_shap_values[0].shape)
    new_shap_values = new_shap_values[0]


    # katz
    # target_names = {'00a11192e6c2286c61bbaf9e8206de75d02a100f6ed7e410c1654fe19c07789b', 'VirusShare_dad40b7991d04253aab542cecf1e6d23',
    #  'cd237dfaff49d19fa6bfebb1b7a0842af73cc634b080afc4fc76277e1228e399',
    #  '9a701b5c58d3aec46863be8e4bf411e1a4a143dd9ffd6062c608e2caf3a7a269', 'VirusShare_58bde35ee25f91c57ed7d3509603d824',
    #  '27e3dbaf27bcd2e313b9239897ef1af24a8472701fcbcc0a65f0ae473db9deed',
    #  '01411844d63ecc8ef3a03069d314ee6f3d4064e7a40cfa9afa90ea74e3e96182',
    #  'd84ffbab7faa8aa63c8f98c52b66dffbd052efc72553b65b8073d30df1f3f39b'}

    # degree
    # target_names ={'VirusShare_b7bec341529aba4f3d0f49388e971926',
# 'VirusShare_58bde35ee25f91c57ed7d3509603d824'}
#     target_names = {'VirusShare_58bde35ee25f91c57ed7d3509603d824'}

    # degree second round
    # target_names = {'47844433ee4c2fbd4fd0dc0880bfbb23cfee7b5baa21146b172d53a63cbe4be0',
    #  '256535ffc9735292e56b44e9dc1a9fbed4ae8e8107dd761546f4015765ea838d', 'VirusShare_5b1b326ab001186e38230594bcedcfa0',
    #  'c0893dbacd2af9946aafe35a011f1ba9fbc7c86cc5a34d5ed3605d55ae80642e'}


    for i in range(start_index, end_index):
        # if '01411844d63ecc8ef3a03069d314ee6f3d4064e7a40cfa9afa90ea74e3e96182' in name[i]:
        #     continue
        fcg_file = name[i].replace('\n', '')
        # true_name = fcg_file.split('/')[-1].split('.gexf')[0]
        # if true_name not in target_names:
        #     continue

        # fcg_file = fcg_file.replace('b/shiwensong/dataset', 'c/shiwensong/Malscan/MalScan-code')
        print("fcg_file", fcg_file)
        load_start = time.time()
        # fcg = FCG(fcg_file, 1, zeros_list)
        # print("shap", new_shap_values[i])
        # fcg = FCG(fcg_file, 1, zeros_list)
        fcg = recover_graph(fcg_file, new_shap_values[i], recover_dir + 'ga_failed/')

        if fcg is None:
            print(fcg_file)
            print("fcg is None")
            continue

        sample_name = fcg_file.split('/')[-1].split('.')[0]
        # print("sample_name", sample_name)
        # failed_graph_dir = f'log_init_200pop_100step_shap/{recover_path}/ga_failed/'
        # fcg = recover_graph(sample_name, zeros_list, failed_graph_dir, 0)

        print("fcg nodes", len(fcg.nodes))
        print("fcg edges", len(fcg.edges))
        print("fcg user nodes", len(fcg.sensitive_user_defined_nodes))
        print("fcg system nodes", len(fcg.sensitive_system_nodes))
        print("fcg sensitive_edges", len(fcg.sensitive_edges))
        print("fcg used_sensitive_nodes", len(fcg.sensitive_nodes))

        if fcg.used_sensitive_nodes is not None and len(fcg.used_sensitive_nodes) != 0:
            begin_time = time.time()
            fcg.cal_centralities(type)
            combined_feature = None
            if type == 'degree':
                combined_feature = fcg.degree_feature
            elif type == 'katz':
                combined_feature = fcg.katz_feature
            elif type == 'closeness':
                combined_feature = fcg.closeness_feature
            elif type == 'harmonic':
                combined_feature = fcg.harmonic_feature
            end_time = time.time()
            print("feature time", end_time - begin_time)
            begin_time = time.time()
            all_samples_features.append(combined_feature)
            # new_shap_value = get_shap(train_dataset, combined_feature, MLP_model)
            # print("new_shap_value", new_shap_value[0].shape)
            # fcg.shap_value = new_shap_value[0]

            combined_feature = np.array(combined_feature)
            combined_feature = combined_feature.reshape(1, -1)

            # all_benign_distance, all_benign_indices = benign_knn_model.kneighbors(combined_feature)
            # all_malware_distance, all_malware_indices = malware_knn_model.kneighbors(combined_feature)
            # benign_distance = (all_benign_distance[0][0] + all_benign_distance[0][1]) / 2.0
            # malware_distance = (all_malware_distance[0][0] + all_malware_distance[0][1]) / 2.0
            y_label = model.predict(combined_feature)
            # score = benign_distance - malware_distance
            MLP_Y_probs = MLP_model.predict(combined_feature)
            MLP_Y_probs2 = MLP_model2.predict(combined_feature)
            end_time = time.time()
            print("test model time", end_time - begin_time)

            print(fcg.apk_name)
            print("y_label", y_label)
            # print("score", score)
            print("MLP_Y_probs", MLP_Y_probs)
            print("MLP_Y_probs2", MLP_Y_probs2)

            if y_label[0] == 0:
                print("benign")

            else:
                print("malware")
                # print(fcg.apk_name)
                ga(fcg, 40, 100, 300, False, None, True, True, False)
                # pop = _init_population(fcg, 5, 300)
                # score_list, mlp_score_list, shap_list = _population_score(fcg, pop)
                # sort_mlp_score_list = sorted(mlp_score_list)
                # print("sort_mlp_score_list", sort_mlp_score_list)
                # target_index = _fitness_function(score_list)
                # if target_index != -1:
                #     print("fitness function is true")
                #     save_log_for_every_individual(fcg, pop[target_index], score_list[target_index], i)

    # all_features = np.array(all_samples_features)
    # new_shap_values = get_shap(train_dataset, all_features, MLP_model2)
    # with open(f'{save_dir}120_samples_{model_name}_{feature_type}.pkl', 'wb') as file:
    #     pickle.dump(new_shap_values, file)
    # 
    # print("new_shap_values", new_shap_values[0].shape)




