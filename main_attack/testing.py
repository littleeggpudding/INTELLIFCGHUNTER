import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import os
sys.path.append(os.path.abspath('../type'))
sys.path.append(os.path.abspath('task'))

from FCG import FCG
import random
import numpy as np
from Mutation import Mutation
from MutationSequence import MutationSequence
from ExtractFeature import obtain_sensitive_apis
from Classify import train_model, test_model,get_distance_from_knn1, get_distance_from_knn10, get_4features_data, get_distance_from_original_knn1
import copy
import json
import psutil
import networkx as nx
import matplotlib.pyplot as plt
import csv
import glob
from networkx.algorithms.isomorphism import GraphMatcher
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import joblib


from MutateFCG import read_log_for_every_generation, read_log_for_last_generation, read_log_for_first_generation

def judge_graph(fcg1, fcg2):
    if len(fcg1.edges) != len(fcg2.edges):
        return False
    if len(fcg1.nodes) != len(fcg2.nodes):
        return False
    for edge in fcg1.edges:
        if edge not in fcg2.edges:
            return False
    for node in fcg1.nodes:
        if node not in fcg2.nodes:
            return False
    return True


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

    return increase_index, decrease_index, orignal_zero, new_zero

def delete_zero_feature(feature):
    new_feature = []
    for i in range(len(feature)):
        if feature[i] != 0:
            new_feature.append(feature[i])
    return new_feature

def generate_csv_data(fcg, mutation, increase_index, decrease_index, orignal_zero, new_zero, score):
    #csv_data = [
    #     ['Node', 'isSystemNode', 'degree', 'katz', 'closeness', 'harmonic', 'in_degree', 'out_degree', 'mutation'],
    # ['node', 'isSystemNode', 'degree', 'katz', 'closeness', 'harmonic', 'in_degree', 'out_degree', 'mutation',
    #  'change_degree', 'change_katz', 'change_closeness', 'change_harmonic', 'score'],
    # ]
    csv_data = []

    increase_node = []


    for node in fcg.nodes:
        is_system_node = 1 if node in fcg.system_nodes else 0
        degree = fcg.degree_feature[fcg.node_index[node]]
        katz = fcg.katz_feature[fcg.node_index[node]]
        closeness = fcg.closeness_feature[fcg.node_index[node]]
        harmonic = fcg.harmonic_feature[fcg.node_index[node]]

        callers = [m for m, n in fcg.edges if n == node and m != n]  # 有的边是自己指向自己的，包括在calleee中，就不用包括在caller中
        callees = [m for n, m in fcg.edges if n == node]

        in_degree = len(callers)
        out_degree = len(callees)



        csv_data.append([node, is_system_node, degree, katz, closeness, harmonic, in_degree, out_degree, mutation])

    return csv_data

def find_caller_callee(fcg, node):
    callers = [m for m, n in fcg.edges if n == node]  # 有的边是自己指向自己的，包括在calleee中，就不用包括在caller中
    callees = [m for n, m in fcg.edges if n == node]
    return callers, callees

def compare_individual(individual, former_individual):
    min_len = min(len(individual), len(former_individual))
    same_cnt = 0
    for j in range(min_len):
        if individual[j] == former_individual[j]:
            same_cnt += 1
    return same_cnt

def analyze_individual(res):
    print("begin to analyze the individual")
    print(len(res))
    for data in res:
        score = eval(data.get("score"))
        # score = data.get("score")
        generation = data.get("generation")
        individual = data.get("individual")
        # print(f"generation: {generation}, score: {score}, individual: {individual}")
        print(f"generation: {generation}, score: {score}")

        # final_fcg = deal_with_sequence(fcg, individual)
        # final_fcg.save('newfcg_random_succeed/')
        # final_fcg.cal_centralities()
        # degree = final_fcg.degree_feature
        # katz = final_fcg.katz_feature
        # closeness = final_fcg.closeness_feature
        # harmonic = final_fcg.harmonic_feature
        # combined_feature = np.hstack((degree, katz, closeness, harmonic))
        # # 存储feature
        # with open('newfcg_random_succeed/' + apk_name + ".pkl", 'wb') as f:
        #     pickle.dump(combined_feature, f)
        #
        # print("end to analyze the individual")

def deal_with_sequence(fcg, individual):
    #individual是一个二维的列表，每个元素是一个列表，表示一个group
    fcg = copy.deepcopy(fcg)
    for i in range(len(individual)):
        group = individual[i]
        # print("group", len(group))
        if len(group) == 0:
            continue
        for mutation in group:
            obj = Mutation(fcg)
            obj.mutation = mutation
            fcg.process_mutation(obj)

    return fcg



def analyze_generation(fcg, data):
    print("begin to analyze the generation")
    # print(res)
    # for i in range(len(res)):
    #     data = res[i]
    score = eval(data.get("score"))
    print("score", len(score))
    # score = data.get("score")
    generation = data.get("generation")
    individual_list = data.get("individual_list")
    print(len(individual_list))
    print("individual 0", len(individual_list[0]))
    shap = data.get("shap")
    idx = np.argsort(score)
    score = sorted(score)
    max_score = score[-1]
    min_score = score[0]

    # 将score和individual_list组合为元组列表
    combined_list = list(zip(score, shap))

    # 根据score对combined_list进行排序
    # combined_list.sort(reverse=True, key=lambda x: x[0])
    # combined_list.sort(key=lambda x: x[0])

    # 解开元组，将排序后的数据重新分配给score和individual_list
    # score, individual_list = zip(*combined_list)

    # print("individual_list", individual_list)
    # if not individual_list is None and generation == 6:
    print(f"generation: {generation}, score: {score},\n shape: {shap},\n")
    print("begin to analyze the last generation")
    # for id in idx:
    #     print("id", id)
    #     print("individual_list", len(individual_list[id]))
    #     final_fcg = deal_with_sequence(fcg, individual_list[id])
    #     if final_fcg is None:
    #         continue
    #     else:
    #         final_fcg.save('newfcg_ga_failed/VirusShare_8a3d12947d7d98dac2e89b51f623bbf0/', id)
    #         final_fcg.cal_centralities()
    #         degree = final_fcg.degree_feature
    #         katz = final_fcg.katz_feature
    #         closeness = final_fcg.closeness_feature
    #         harmonic = final_fcg.harmonic_feature
    #         combined_feature = np.hstack((degree, katz, closeness, harmonic))
    #         #存储feature
    #         with open('newfcg_ga_failed/VirusShare_8a3d12947d7d98dac2e89b51f623bbf0/'+str(id)+".pkl", 'wb') as f:
    #             pickle.dump(combined_feature, f)
    #
    #     print("end to analyze the last generation")

    return min_score, max_score, shap[0]
    



if __name__ == '__main__':
    # file_path = "VirusShare_803a95b3c94e8dd599c824331c9a7e55"
    # file_path = 'VirusShare_038d756ffdb66cf297fb4bbc6cd994ba_log_for_every_individual_430features_MLP.json'#分析
    # file_path = 'VirusShare_fe35da9300d8373b59ee24ca3f33c354_log_for_every_individual_430features.json'

    #analyze the last generation
    # dir_path = '/data/c/shiwensong/malwareGA/task/log_init_200pop_100step_shap/jan24_MLP_test_ga_sensitive_area_all/VirusShare_6a9b2fac202d0ab867f89cd79c17f5c5_log_for_every_generation.json' #malscan 2017

    # apk_name = dir_path.split('/')[-1].split('_')[0] + '_' + dir_path.split('/')[-1].split('_')[1]
    # print(apk_name)
    #
    # fcg_dir = '/data/c/shiwensong/Malscan/MalScan-code/virusshare2018_gexf/'
    # fcg_file = fcg_dir + apk_name + '.gexf'

    # read shap value
    # with open('shap_values_30_samples.pkl', 'rb') as f:
    #     shap_values = pickle.load(f)
    #
    # shap_values = shap_values[0][26]

    # fcg = FCG(fcg_file, 1, shap_values)
    fcg = None

    # fcg.cal_centralities()
    # degree = fcg.degree_feature
    # katz = fcg.katz_feature
    # closeness = fcg.closeness_feature
    # harmonic = fcg.harmonic_feature
    # combined_feature = np.hstack((degree, katz, closeness, harmonic))
    # combined_feature = combined_feature.reshape(1, -1)
    # pros = test_model(combined_feature, 1, 'MLP', model_path='./430features_3yearsdataset_all/MLP.h5')
    # print(pros[0][0])


    # res = read_log_for_every_generation(dir_path)
    # print("res", len(res))
    # #
    # analyze_generation(fcg, res)

    #analyze the success individual

    # path = 'log_init_200pop_100step_shap/feb27_MLP_test_ga_dominate_shap_score_forth_ga_top1/' #malscan 2017
    path = '/data/c/shiwensong/project/test_mlp_substitute_with_shap/RandomForest/degree/'


    # files = glob.glob(path + '/*_log_for_every_generation.json')
    files = [path+"e8aefdd6d3770a690f7d99209f3bf7205ab3195410fddca85d3892a97193411f_log_for_every_generation.json"]
    for file in files:
        all_generations = read_log_for_every_generation(file)
        print("--------------")
        print("file", file)
        print("all_generations", len(all_generations))
        for i in range(len(all_generations)):
            generation = all_generations[i]
            min_score, max_score,shap = analyze_generation(fcg, generation)
            print(f"i = {i}, min_score = {min_score}, max_score = {max_score}, {shap}")


    # #
    # all_success_files = glob.glob(path + '/*_log_for_every_individual.json')
    # all_success_file_names = []
    # #
    # for file in all_success_files:
    #     file_name = file.split('/')[-1].split('_')[0] + '_' + file.split('/')[-1].split('_')[1]
    #     # all_success_file_names.append(file_name)
    #     res = read_log_for_every_generation(file)
    #     analyze_individual(res)
    # #


    #
    # model = joblib.load('./430features_3yearsdataset_all/AdaBoost.pkl')
    #
    # for dir_path in files:
    #     print("dir_path", dir_path)
    #
    #     apk_name = dir_path.split('/')[-1].split('_')[0] + '_' + dir_path.split('/')[-1].split('_')[1]
    #     print(apk_name)
    #
    #     # fcg_dir = '/data/c/shiwensong/Malscan/MalScan-code/virusshare2018_gexf/'
    #     fcg_dir = '/data/b/shiwensong/dataset/virusshare2018_gexf/'
    #     fcg_file = fcg_dir + apk_name + '.gexf'
    #
    #     fcg = FCG(fcg_file, 1)
    #     fcg.cal_centralities()
    #     degree = fcg.degree_feature
    #     katz = fcg.katz_feature
    #     closeness = fcg.closeness_feature
    #     harmonic = fcg.harmonic_feature
    #     combined_feature = np.hstack((degree, katz, closeness, harmonic))
    #     combined_feature = combined_feature.reshape(1, -1)
    #     # pros = test_model(combined_feature, 1, 'MLP', model_path='./430features_3yearsdataset_all/MLP.h5')
    #     # print(pros[0][0])
    #     pros = model.predict_proba(combined_feature)
    #
    #     all_generations = read_log_for_every_generation(dir_path)
    #
    #     for i in range(len(all_generations)):
    #         generation = all_generations[i]
    #         min_score, max_score = analyze_generation(fcg, generation)
    #         print(f"i = {i}, min_score = {min_score}, max_score = {max_score}")




    #     first_generation = read_log_for_first_generation(dir_path)
    #     last_generation = read_log_for_last_generation(dir_path)
    # 
    # 
    #     # analyze_individual(fcg, res, apk_name)
    #     print("first_generation", len(first_generation))
    #     first_generation_min_score, first_generation_max_score = analyze_generation(fcg, first_generation)
    # 
    #     print("last_generation", len(last_generation))
    #     last_generation_min_score, last_generation_max_score = analyze_generation(fcg, last_generation)
    # 
    #     is_success = apk_name in all_success_file_names
    # 
    #     data = [
    #         apk_name,
    #         pros[0][1],
    #         first_generation_min_score,
    #         first_generation_max_score,
    #         last_generation_min_score,
    #         last_generation_max_score,
    #         is_success
    #     ]
    #     res.append(data)
    # 
    # print("res", len(res))
    # df = pd.DataFrame(res, columns=['apk_name', 'init score', 'first_generation_min_score', 'first_generation_max_score', 'last_generation_min_score', 'last_generation_max_score', 'is_success'])
    # df.to_csv('ga_shap_score_forth_ga_top1.csv', index=False)





    # file_path = glob.glob(dir_path + '*for_every_individual.json')
    # length = len(file_path)
    # for i in range(length):
    #     file = file_path[i]
    #
    #     res = read_log_for_every_generation(file)
    #
    #     print(file)
    #     # analyze_generation(res)
    #     analyze_individual(res)

    #analyze the how many generation
    # dir_path = '/data/c/shiwensong/malwareGA/task/log_init_200pop_100step_shap/jan24_MLP_test_ga_sensitive_area_all'
    # file_path = glob.glob(dir_path + '/*_log_for_every_generation.json')
    # length = len(file_path)
    # for i in range(length):
    #     file = file_path[i]
    #
    #     res = read_log_for_every_generation(file)
    #
    #     print(file)
    #     # analyze_generation(res)
    #     analyze_generation(res)




    # print("test")

    #37 264 284 这三个节点

    #只记录sensitive api
    # csv_data = [
    #     ['score', 'degree', 'degree_increase_index', 'degree_decrease_index', 'katz', 'katz_increase_index', 'katz_decrease_index', 'closeness', 'closeness_increase_index', 'closeness_decrease_index', 'harmonic', 'harmonic_increase_index', 'closeness_decrease_index', 'mutation'],
    # ]
    #
    # #check invidivual
    # for data in res:
    #     # score = eval(data.get("score"))
    #     score = data.get("score")
    #     generation = data.get("generation")
    #     individual = data.get("individual")
    #     print(f"generation: {generation}, score: {score}, individual: {individual}")
    #     # if score is not None:
    #     #     score.sort(reverse=True)
    #     print(f"generation: {generation}, score: {score}, individual: {individual}")
    #     print("------------------")
    #     fcg = FCG('/data/c/shiwensong/Malscan/MalScan-code/virusshare2018_gexf/VirusShare_63acec04855ac0c5641247f5ba3d48b9.gexf', 1)
    #     fcg.cal_centralities()
    #     # print("observed nodes")
    #     # node_name1 = sensitive_apis[37]
    #     # node_name2 = sensitive_apis[264]
    #     # node_name3 = sensitive_apis[284]
    #     # print(node_name1)
    #     # callers, callees = find_caller_callee(fcg, node_name1)
    #     # print(callers, callees)
    #     # print(node_name2)
    #     # callers, callees = find_caller_callee(fcg, node_name2)
    #     # print(callers, callees)
    #     # print(node_name3)
    #     # callers, callees = find_caller_callee(fcg, node_name3)
    #     # print(callers, callees)
    #
    #
    #     print("fcg.edges", len(fcg.edges))
    #     print("fcg.nodes", len(fcg.nodes))
    #     print("fcg system nodes", fcg.system_nodes)
    #     print("fcg degree_feature", fcg.degree_feature)
    #     print("fcg katz_feature", fcg.katz_feature)
    #     print("fcg closeness_feature", fcg.closeness_feature)
    #     print("fcg harmonic_feature", fcg.harmonic_feature)
    #
    #     # 将节点和对应的"isSystemNode"值存储在一个字典中
    #     # node_system_dict = {node: 1 if node in fcg.system_nodes else 0 for node in fcg.nodes}
    #     #
    #     # 将信息写入CSV文件
    #     # with open('gexfTest.csv', mode='w', newline='') as csv_file:
    #     #     fieldnames = ['Node', 'isSystemNode']  # 列名
    #     #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     #
    #     #     writer.writeheader()  # 写入列名
    #     #
    #     #     # 写入每个节点和对应的"isSystemNode"值
    #     #     for node, is_system_node in node_system_dict.items():
    #     #         writer.writerow({'Node': node, 'isSystemNode': is_system_node})
    #     #
    #     # original
    #     print("original")
    #     # 计算四种feature
    #     former_degree_feature = fcg.degree_feature
    #     former_katz_feature = fcg.katz_feature
    #     former_closeness_feature = fcg.closeness_feature
    #     former_harmonic_feature = fcg.harmonic_feature
    #     combined_feature = np.concatenate((former_degree_feature, former_katz_feature, former_closeness_feature, former_harmonic_feature), axis=0)
    #     combined_feature = combined_feature.reshape(1, -1)
    #     # 分别用各种分类器测试
    #     # benign_distance, malware_distance = get_distance_from_knn1(combined_feature)
    #     # print(f"benign_distance: {benign_distance}, malware_distance: {malware_distance}")
    #     # print(benign_distance-malware_distance)
    #     Y_pred_probs = test_model(combined_feature, 1, 'MLP', model_path='./430features_3yearsdataset_all/MLP.h5')
    #     print(Y_pred_probs[0][0])
    #     #mutation
    #     print("mutation")
    #     print(len(individual))
    #     cnt = 0
    #     for i in range(len(individual)):
    #         group = individual[i]
    #         print("group", len(group))
    #         print("------------------")
    #         for mutation in group:
    #             obj = Mutation(fcg)
    #             obj.mutation = mutation
    #             print(obj.mutation)
    #             state = fcg.process_mutation(obj)
    #             print("state", state)
    #             print("operation")
    #             # print(mutation)
    #             print("+++++++++++++++++++++++++++++++")
    #
    #             # 计算四种feature
    #             fcg.cal_centralities()
    #             degree_feature = fcg.degree_feature
    #             katz_feature = fcg.katz_feature
    #             closeness_feature = fcg.closeness_feature
    #             harmonic_feature = fcg.harmonic_feature
    #
    #             print("check feature change")
    #             check_degree = check_feature(former_degree_feature, degree_feature)
    #             check_katz = check_feature(former_katz_feature, katz_feature)
    #             check_closeness = check_feature(former_closeness_feature, closeness_feature)
    #             check_harmonic = check_feature(former_harmonic_feature, harmonic_feature)
    #
    #             # print(node_name1)
    #             # callers, callees = find_caller_callee(fcg, node_name1)
    #             # print(callers, callees)
    #             # print(node_name2)
    #             # callers, callees = find_caller_callee(fcg, node_name2)
    #             # print(callers, callees)
    #             # print(node_name3)
    #             # callers, callees = find_caller_callee(fcg, node_name3)
    #             # print(callers, callees)
    #
    #
    #             combined_feature = np.concatenate((degree_feature, katz_feature, closeness_feature, harmonic_feature),
    #                                               axis=0)
    #             combined_feature = combined_feature.reshape(1, -1)
    #             # 分别用各种分类器测试
    #             Y_pred_probs = test_model(combined_feature, 1, 'MLP',
    #                                               model_path='./430features_3yearsdataset_all/MLP.h5')
    #             print(Y_pred_probs[0][0])
    #
    #             # csv_data.append([Y_pred_probs[0][0], delete_zero_feature(degree_feature), check_degree[0], check_degree[1], delete_zero_feature(katz_feature), check_katz[0], check_katz[1], delete_zero_feature(closeness_feature), check_closeness[0], check_closeness[1], delete_zero_feature(harmonic_feature), check_harmonic[0], check_harmonic[1], mutation])
    #             #
    #             # former_degree_feature = degree_feature
    #             # former_katz_feature = katz_feature
    #             # former_closeness_feature = closeness_feature
    #             # former_harmonic_feature = harmonic_feature
    #
    #             if Y_pred_probs < 0:
    #                 print("i", i)
    #                 print("success")
    #                 fcg.save('newfcg/')

    # print("csv_data", len(csv_data))

    # file_name = 'analysis_one_success_process/F8626CED9A5689B029E441C74AEACA4C81DE3FEE8B11795370C37BB5AED00F04.csv'
    # with open(file_name, 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerows(csv_data)

    #check generation
    # print(len(res))

    # # #

    #check graph
    # fcg1 = FCG('/data/c/shiwensong/Malscan/MalScan-code/virusshare2018_gexf/VirusShare_7ea5fcbbcfe86699499af3daefeecc14.gexf', 1)
    # fcg2 = FCG('/data/c/shiwensong/Malscan/MalScan-code/virusshare2018_gexf/VirusShare_7ec286d61ee68c89f4cbfae9c5a54920.gexf', 1)
    # fcg3 = FCG('/data/c/shiwensong/Malscan/MalScan-code/virusshare2018_gexf/VirusShare_7ed121e9d7faab04e5dcf2e70ca3b648.gexf', 1)
    # check1 = judge_graph(fcg1, fcg2)
    # check2 = judge_graph(fcg1, fcg3)
    # check3 = judge_graph(fcg2, fcg3)
    # print(len(fcg1.edges))
    # print(len(fcg2.edges))
    # print(len(fcg3.edges))
    # print(len(fcg1.nodes))
    # print(len(fcg2.nodes))
    # print(len(fcg3.nodes))
    # print(check1)
    # print(check2)
    # print(check3)