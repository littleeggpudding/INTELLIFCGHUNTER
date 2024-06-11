import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pickle
import shap
import numpy as np
import joblib
import pandas as pd
import os
import sys
from Classify import test_dataset_benign, test_dataset_malware
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath('../type'))
sys.path.append(os.path.abspath('task'))
from FCG import FCG
import time
import random
import copy
from ExtractFeature import obtain_sensitive_apis
from Classify import obtain_one_year_dataset_4features
import torch

def obtain_dataset(year= '2018'):

    X_train_benign, Y_train_benign = obtain_one_year_dataset_4features('/data/b/shiwensong/dataset/feature_Nov30/', year, '0')
    X_train_malware, Y_train_malware = obtain_one_year_dataset_4features('/data/b/shiwensong/dataset/feature_Nov30/', year, '1')
    print("X_train_benign", len(X_train_benign))

    print("X_train_malware", len(X_train_malware))

    X_train = np.vstack((X_train_benign, X_train_malware))
    print("X_train", X_train.shape)
    return X_train
def load_model(model_path):
    #加载模型
    if 'MLP' in model_path:
        MLP_model = tf.keras.models.load_model(model_path)
        return MLP_model
    else:
        model = joblib.load(model_path)
        return model

def attack_feature(shap_value, feature, MLP_model, rate = 0.01):
    print("shap value", shap_value.shape)
    print("feature", feature.shape)

    neg_shap_value = []
    pos_shap_value = []
    for i in range(1720):
        if shap_value[i] < 0:
            neg_shap_value.append(i)
        elif shap_value[i] > 0:
            pos_shap_value.append(i)

    print("neg_shap_value", len(neg_shap_value))
    print("pos_shap_value", len(pos_shap_value))

    # 正向的特征减小
    former_pros = 2
    for i in range(len(pos_shap_value)):
        idx = pos_shap_value[i]
        if feature[0][idx] != 0:
            feature[0][idx] = feature[0][idx] - feature[0][idx] * rate
            print("feature[0][idx]", feature[0][idx])
            pros = MLP_model.predict_proba(feature)
            # if pros[0][1] > former_pros:
            #     feature[0][idx] = feature[0][idx] + feature[0][idx] * rate

            former_pros = pros[0][1]

            print("pos pros", pros)
            if pros[0][1] < 0.5:
                print("success")
                return feature


    #负向的特征增大
    for i in range(len(neg_shap_value)):
        idx = neg_shap_value[i]
        if feature[0][idx] != 0:
            feature[0][idx] = feature[0][idx] + feature[0][idx] * rate
            print("feature[0][idx]", feature[0][idx])
            pros = MLP_model.predict_proba(feature)
            # if pros[0][0] > former_pros:
            #     feature[0][idx] = feature[0][idx] - feature[0][idx] * rate

            former_pros = pros[0][1]
            print("neg pros", pros)
            if pros[0][1] < 0.5:
                print("success")
                return feature

    return None

def attack_feature_new(shap_value, feature, MLP_model, rate = 0.01):
    print("shap value", shap_value.shape)
    print("feature", feature.shape)

    neg_shap_value = []
    pos_shap_value = []
    non_zero_shape_value = []
    for i in range(1720):
        if shap_value[i] < 0:
            neg_shap_value.append(shap_value[i])
            non_zero_shape_value.append(shap_value[i])
        elif shap_value[i] > 0:
            pos_shap_value.append(shap_value[i])
            non_zero_shape_value.append(shap_value[i])


    print("neg_shap_value", len(neg_shap_value))
    print("pos_shap_value", len(pos_shap_value))

    # 使用sorted函数按照绝对值排序，并返回下标
    sorted_indices = sorted(range(len(non_zero_shape_value)), key=lambda x: abs(non_zero_shape_value[x]), reverse=True)


    # 正向的特征减小
    former_pros = 2
    cnt = 0

    for idx in sorted_indices:
        cur_shap_value = non_zero_shape_value[idx]
        cur_feature = feature[0][idx]
        if cur_feature != 0 and cur_shap_value != 0:
            if cur_shap_value > 0:
                while True:
                    if cur_feature - rate < 0:
                        break
                    cur_feature = cur_feature - rate
                    feature[0][idx] = cur_feature
                    pros = MLP_model.predict(feature)
                    if pros[0][0] > former_pros:
                        feature[0][idx] = cur_feature + rate
                    else:
                        former_pros = pros[0][0]
                        cnt += 1

                    print("pos pros", pros)
                    if former_pros < 0.5:
                        print("success")
                        return feature

            else:
                try_times = 0
                while try_times < 1000:
                    try_times += 1
                    cur_feature = cur_feature + rate
                    feature[0][idx] = cur_feature
                    pros = MLP_model.predict(feature)
                    if pros[0][0] > former_pros:
                        feature[0][idx] = cur_feature - rate
                    else:
                        former_pros = pros[0][0]
                        cnt += 1

                    print("neg pros", pros)
                    if former_pros < 0.5:
                        print("success")
                        return feature

    print("cnt", cnt)

    return None



def get_shap(X_train, X_test, MLP_model):
    # X_train = np.array(X_train)
    # print("X_train", X_train.shape)
    # mean_background = np.mean(X_train, axis=0).reshape(1, -1)  # 确保它是二维的
    # # change to tensor
    # mean_background = torch.tensor(mean_background, dtype=torch.float32)
    # X_train = X_train[:1]
    #give a test dataset, return the shap value of each sample
    explainer = shap.KernelExplainer(MLP_model.predict, X_train)
    # for i in range(5):
    #     shap_values = explainer.shap_values(X_test[i])
    #     print("shap_values", shap_values)
    #     # free gpu
    #     tf.keras.backend.clear_session()
    shap_values = explainer.shap_values(X_test, gc_collect=True)
    return shap_values

def get_explainer_shap(X_test, MLP_model, num_features):
    #give a test dataset, return the shap value of each sample
    explainer = shap.DeepExplainer(MLP_model.predict, torch.zeros((1, num_features)))
    # for i in range(5):
    #     shap_values = explainer.shap_values(X_test[i])
    #     print("shap_values", shap_values)
    #     # free gpu
    #     tf.keras.backend.clear_session()
    shap_values = explainer.shap_values(X_test, gc_collect=True)
    return shap_values


def get_shap_probs(X_train, X_test, model):
    #give a test dataset, return the shap value of each sample
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    shap_values = explainer.shap_values(X_test)
    return shap_values

def get_shap_other_tree_explain(X_test, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values


def get_PermutationExplainer(X_test, model):
    """
    Compute SHAP values using PermutationExplainer for a given model and test dataset.

    Args:
        X_test (np.array or pd.DataFrame): The test dataset used to compute SHAP values.
        model (Any): A trained machine learning model compatible with SHAP's explainer.

    Returns:
        shap_values (SHAP values object): Computed SHAP values for the test dataset.
    """
    # 创建一个掩码函数，用于模型的输入。这里我们直接使用数据本身作为背景。
    masker = shap.maskers.Independent(data=X_test)

    # 初始化 PermutationExplainer，传入模型和掩码函数
    explainer = shap.PermutationExplainer(model.predict, masker)

    # 计算 SHAP 值
    shap_values = explainer(X_test)

    return shap_values



def random_m(fcg, steps):
    # print("random_m starting")
    fcg = copy.deepcopy(fcg)
    # 将给定的fcgs随机变换若干步,并保存
    # 获取一个 apk 的 call graph
    # print("fcg nodes", len(fcg.nodes))
    # print("fcg edges", len(fcg.edges))

    # 每次传入是一个新的fcg

    # print("steps", steps)

    mutation_list = []
    visited_random_number = set()

    cnt_random_1 = 0  # add node 1
    cnt_random_2 = 0  # add node 2
    cnt_random_3 = 0  # add edge 1
    cnt_random_4 = 0  # add edge 2
    # cnt_random_5 = 0 # insert node
    # cnt_random_5 = 0 #add edge 3
    # cnt_random_6 = 0 #add edge 4
    # cnt_random_7 = 0 #rewiring 1
    # cnt_random_8 = 0  # rewiring 2
    # cnt_random_9 = 0 #remove node
    # cnt_random_10 = 0#remove node

    count = set()

    i = 0
    while i < steps:
        # 生成随机数1-4
        # print('iteration', i)
        # print("mutation ing edges", len(mutation.edges), len(fcg.edges))
        # print("mutation ing nodes", len(mutation.nodes), len(fcg.nodes))
        random_int = random.randint(1, 4)
        # print("random_int", random_int)

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
            continue
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

def analyze(sample_id, shap_vector1, shap_vector2, shap_vector3, shap_vector4, origin_feature=None):
    #每一个都是1720长度的
    #排序
    shap_vector1_neg = []
    shap_vector1_pos = []
    shap_vector2_neg = []
    shap_vector2_pos = []
    shap_vector3_neg = []
    shap_vector3_pos = []
    shap_vector4_neg = []
    shap_vector4_pos = []
    for i in range(1720):

        if shap_vector1[i] < 0:
            shap_vector1_neg.append(shap_vector1[i])
        elif shap_vector1[i] > 0:
            shap_vector1_pos.append(shap_vector1[i])

        if shap_vector2[i] < 0:
            shap_vector2_neg.append(shap_vector2[i])
        elif shap_vector2[i] > 0:
            shap_vector2_pos.append(shap_vector2[i])

        if shap_vector3[i] < 0:
            shap_vector3_neg.append(shap_vector3[i])
        elif shap_vector3[i] > 0:
            shap_vector3_pos.append(shap_vector3[i])

        if shap_vector4[i] < 0:
            shap_vector4_neg.append(shap_vector4[i])
        elif shap_vector4[i] > 0:
            shap_vector4_pos.append(shap_vector4[i])

    print("length of shap_vector1_neg", len(shap_vector1_neg))
    print("length of shap_vector1_pos", len(shap_vector1_pos))
    print("length of shap_vector2_neg", len(shap_vector2_neg))
    print("length of shap_vector2_pos", len(shap_vector2_pos))
    print("length of shap_vector3_neg", len(shap_vector3_neg))
    print("length of shap_vector3_pos", len(shap_vector3_pos))
    print("length of shap_vector4_neg", len(shap_vector4_neg))
    print("length of shap_vector4_pos", len(shap_vector4_pos))

    #排序，得到下标
    idx1_neg = np.argsort(shap_vector1_neg)#从小到大，越小越靠近benign
    idx1_pos = np.argsort(shap_vector1_pos)[::-1]#从大到小，越大越靠近malware
    idx2_neg = np.argsort(shap_vector2_neg)#从小到大
    idx2_pos = np.argsort(shap_vector2_pos)[::-1]#从大到小
    idx3_neg = np.argsort(shap_vector3_neg)#从小到大
    idx3_pos = np.argsort(shap_vector3_pos)[::-1]#从大到小
    idx4_neg = np.argsort(shap_vector4_neg)#从小到大
    idx4_pos = np.argsort(shap_vector4_pos)[::-1]#从大到小

    for i in range(10):
        print("idx1_neg", idx1_neg[i], shap_vector1_neg[idx1_neg[i]])
        print("idx1_pos", idx1_pos[i], shap_vector1_pos[idx1_pos[i]])
        print("idx2_neg", idx2_neg[i], shap_vector2_neg[idx2_neg[i]])
        print("idx2_pos", idx2_pos[i], shap_vector2_pos[idx2_pos[i]])
        print("idx3_neg", idx3_neg[i], shap_vector3_neg[idx3_neg[i]])
        print("idx3_pos", idx3_pos[i], shap_vector3_pos[idx3_pos[i]])
        print("idx4_neg", idx4_neg[i], shap_vector4_neg[idx4_neg[i]])
        print("idx4_pos", idx4_pos[i], shap_vector4_pos[idx4_pos[i]])
        print("===============")


    print("analyze all features")

    shap_sum = 0
    cnt_same_direction = 0
    cnt_diff_direction = 0
    cnt_small_diff = 0
    cnt_total = 0
    for i in range(1720):
        shap_sum = shap_sum + shap_vector1[i]
        #分析 random 100
        #1. direction
        if shap_vector1[i] != 0:
            cnt_total += 1

            if shap_vector1[i] * shap_vector2[i] > 0:
                cnt_same_direction += 1
            elif shap_vector1[i] * shap_vector2[i] < 0:
                cnt_diff_direction += 1
            if shap_vector2[i]==0 and abs(shap_vector1[i] - shap_vector2[i]) < 0.005:
                cnt_small_diff += 1

        #2. diff


        # if shap_vector1[i] < 0 and shap_vector2[i] < 0 and shap_vector3[i] < 0 and shap_vector4[i] < 0:
        #     print("negative", sample_id)
        #     print("i", i)
        #     print("shap_vector1", shap_vector1[i])
        #     print("shap_vector2", shap_vector2[i])
        #     print("shap_vector3", shap_vector3[i])
        #     print("shap_vector4", shap_vector4[i])
        #     print("===============")
        # elif shap_vector1[i] > 0 and shap_vector2[i] > 0 and shap_vector3[i] > 0 and shap_vector4[i] > 0:
        #     print("positive", sample_id)
        #     print("i", i)
        #     print("shap_vector1", shap_vector1[i])
        #     print("shap_vector2", shap_vector2[i])
        #     print("shap_vector3", shap_vector3[i])
        #     print("shap_vector4", shap_vector4[i])
        #     print("===============")
        # elif shap_vector1[i] != 0:#至少一个不为0的
        #     print("special", sample_id)
        #     print("i", i)
        #     print("shap_vector1", shap_vector1[i])
        #     print("shap_vector2", shap_vector2[i])
        #     print("shap_vector3", shap_vector3[i])
        #     print("shap_vector4", shap_vector4[i])
        #     print("===============")
        #
        #     diff1 = shap_vector2[i] - shap_vector1[i]
        #     diff2 = shap_vector3[i] - shap_vector1[i]
        #     diff3 = shap_vector4[i] - shap_vector1[i]
        #
        #     if shap_vector1[i] * shap_vector2[i] >= 0:
        #         print("same direction random 100")
        #
        #     if shap_vector1[i] * shap_vector3[i] >= 0:
        #         print("same direction random 500")
        #
        #     if shap_vector1[i] * shap_vector4[i] >= 0:
        #         print("same direction random 300")
        #
        #     if abs(diff1) < 0.005:
        #         print("diff1 < 0.005")
        #         print("random 100")
        #
        #     if abs(diff2) < 0.005:
        #         print("diff2 < 0.005")
        #         print("random 500")
        #
        #     if abs(diff3) < 0.005:
        #         print("diff3 < 0.005")
        #         print("random 300")


            # if shap_vector1[i] * shap_vector2[i] * shap_vector4[i]> 0:
            #     print("random 100, 300")
            #
            # if shap_vector1[i] * shap_vector3[i] * shap_vector4[i]> 0:
            #     print("random 500, 300")

    # print("shap_sum", shap_sum)
    print("cnt_same_direction", cnt_same_direction)
    print("cnt_diff_direction", cnt_diff_direction)
    print("cnt_small_diff", cnt_small_diff)
    print("cnt_total", cnt_total)


    # print("analyze 4 features")
    # degree = origin_feature[:430]
    # print("degree", degree.shape)
    # katz = origin_feature[430:860]
    # closeness = origin_feature[860:1290]
    # harmonic = origin_feature[1290:]
    #
    # shap_vector1_degree = shap_vector1[:430]
    # shap_vector1_katz = shap_vector1[430:860]
    # shap_vector1_closeness = shap_vector1[860:1290]
    # shap_vector1_harmonic = shap_vector1[1290:]
    #
    # shap_vector2_degree = shap_vector2[:430]
    # shap_vector2_katz = shap_vector2[430:860]
    # shap_vector2_closeness = shap_vector2[860:1290]
    # shap_vector2_harmonic = shap_vector2[1290:]
    #
    # shap_vector3_degree = shap_vector3[:430]
    # shap_vector3_katz = shap_vector3[430:860]
    # shap_vector3_closeness = shap_vector3[860:1290]
    # shap_vector3_harmonic = shap_vector3[1290:]
    #
    # shap_vector4_degree = shap_vector4[:430]
    # shap_vector4_katz = shap_vector4[430:860]
    # shap_vector4_closeness = shap_vector4[860:1290]
    # shap_vector4_harmonic = shap_vector4[1290:]
    #
    # for i in range(430):
    #     shap_vector1_feature = degree[i] * shap_vector1_degree[i] + katz[i] * shap_vector1_katz[i] + closeness[i] * shap_vector1_closeness[i] + harmonic[i] * shap_vector1_harmonic[i]
    #     shap_vector2_feature = degree[i] * shap_vector2_degree[i] + katz[i] * shap_vector2_katz[i] + closeness[i] * shap_vector2_closeness[i] + harmonic[i] * shap_vector2_harmonic[i]
    #     shap_vector3_feature = degree[i] * shap_vector3_degree[i] + katz[i] * shap_vector3_katz[i] + closeness[i] * shap_vector3_closeness[i] + harmonic[i] * shap_vector3_harmonic[i]
    #     shap_vector4_feature = degree[i] * shap_vector4_degree[i] + katz[i] * shap_vector4_katz[i] + closeness[i] * shap_vector4_closeness[i] + harmonic[i] * shap_vector4_harmonic[i]
    #
    #     if shap_vector1_feature > 0 and shap_vector2_feature > 0 and shap_vector3_feature > 0 and shap_vector4_feature > 0:
    #         print("positive")
    #         print("i", i)
    #         print("shap_vector1_feature", shap_vector1_feature)
    #         print("shap_vector2_feature", shap_vector2_feature)
    #         print("shap_vector3_feature", shap_vector3_feature)
    #         print("shap_vector4_feature", shap_vector4_feature)
    #         print("===============")
    #
    #     elif shap_vector1_feature < 0 and shap_vector2_feature < 0 and shap_vector3_feature < 0 and shap_vector4_feature < 0:
    #         print("negative")
    #         print("i", i)
    #         print("shap_vector1_feature", shap_vector1_feature)
    #         print("shap_vector2_feature", shap_vector2_feature)
    #         print("shap_vector3_feature", shap_vector3_feature)
    #         print("shap_vector4_feature", shap_vector4_feature)
    #         print("===============")
    #
    #     elif shap_vector1_feature != 0 or shap_vector2_feature != 0 or shap_vector3_feature != 0 or shap_vector4_feature != 0:
    #         print("special")
    #         print("shap_vector1_feature", shap_vector1_feature)
    #         print("shap_vector2_feature", shap_vector2_feature)
    #         print("shap_vector3_feature", shap_vector3_feature)
    #         print("shap_vector4_feature", shap_vector4_feature)
    #
    #         if shap_vector1_feature > 0 and shap_vector2_feature > 0:
    #             print("random 100")
    #         elif shap_vector1_feature < 0 and shap_vector2_feature < 0:
    #             print("random 100")
    #
    #     elif shap_vector1_feature == 0 and shap_vector2_feature == 0 and shap_vector3_feature == 0 and shap_vector4_feature > 0:
    #         print("random 300")
    #
    #     print("===============")



def ga_optimize(feature, shap_values, model, apk_name, mutation_rate=0.1, generations=100):
    feature = feature[0]
    vector_length = len(feature)

    def evaluate_fitness(vector):
        modified_feature = feature.copy()
        for i in range(vector_length):
            if vector[i] != 0:
                change = vector[i] * shap_values[i] * 0.01
                modified_feature[i] += change
                modified_feature[i] = max(0, modified_feature[i])

        confidence = model.predict(modified_feature.reshape(1, -1))[0][0]
        fitness = -1 if confidence >= 0.5 else -confidence  # 更加关注置信度降低
        return fitness, modified_feature

    population_size = 100  # 可考虑增加种群大小
    population = [np.random.uniform(-1, 1, vector_length) for _ in range(population_size)]
    best_fitness = float('-inf')
    best_individual = None

    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            fitness, modified_feature = evaluate_fitness(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = (individual, modified_feature)
            fitness_scores.append(fitness)

        num_parents = population_size // 2
        parents_indices = np.argsort(fitness_scores)[::-1][:num_parents]
        parents = [population[i] for i in parents_indices]

        children = []
        while len(children) < population_size:
            parent1, parent2 = random.choice(parents), random.choice(parents)
            crossover_point = random.randint(1, vector_length - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            for i in range(vector_length):
                if random.random() < mutation_rate:
                    child[i] += random.uniform(-0.01, 0.01)
                    child[i] = max(0, child[i])
            children.append(child)

        population = parents + children

    if best_individual:
        optimized_feature = best_individual[1]
        # 存储
        with open('attack_feature_ga/' + apk_name + 'optimized_feature.pkl', 'wb') as f:
            pickle.dump(optimized_feature, f)

        with open("attack_feature_ga/score.txt", 'a') as f:
            f.write(apk_name + '\t' + str(best_fitness) + '\n')

        return optimized_feature, best_fitness
    else:
        return None, None



if __name__ == '__main__':
    # train_dataset_2018 = obtain_dataset()
    # train_dataset_2022 = obtain_dataset('2022')
    # train_dataset_2023 = obtain_dataset('2023')
    # train_dataset = np.vstack((train_dataset_2018, train_dataset_2022, train_dataset_2023))
    # print("train_dataset", train_dataset.shape)
    #
    # MLP_model = load_model('./430features_6yearsdataset_1000samples//all_usxRandomForest.pkl')

    #read sample name
    # with open("random_1000_samples.txt", "r") as f:
    #     samples = f.readlines()

    # print("len(samples)", len(samples))
    # print("samples[0]", samples[0])

    # name = [
    #     'VirusShare_5e1d0b712f856e0594ccceee2c2135e7',
    #     "VirusShare_836a62bec037576e17d16bb1bd036ffb",
    #     "VirusShare_45e67a88d73488396eae77a7309d90e7",
    #     "VirusShare_57657cc2a67c756100ef2b5c55dd47ba",
    #     "VirusShare_c508d918a080b807fd23b51c350f86a5",
    #     "VirusShare_9b0d37e5cb949ab86354788ce5488375",
    #     "VirusShare_574e59fd51e7e894b296f684eaa37356",
    #     "VirusShare_4836ea8ec7adc537d17d1357caedf305",
    #     "VirusShare_160b455cc19e0c1ba1bef1b3a9ca97ce",
    #     "VirusShare_0b2d190d17d50dfd4a589aca1a9caa49",
    #     "VirusShare_4a4e5889fedccbe245be4489ce07dff1",
    #     "VirusShare_0d88318b0186b5d65b87899dbaa7a5b3",
    #     "VirusShare_2a944cca05ce869d504d2f1a15f66140",
    #     "VirusShare_f3ddd6fc9631a8679c1adefc3fa778b9",
    #     "VirusShare_28a2725940ecb8b9686476cfa0dda209",
    #     "VirusShare_b4ca4cd07d5e87821012bdaa1faa2096",
    #     "VirusShare_d59b8493b1ad4b600051c5c28d4b52b5",
    #     "VirusShare_62f35131cf856d486a3433c9b94f8200",
    #     "VirusShare_7f1d201c88fa16e39ea198fbc5b99553",
    #     "VirusShare_6ceede843c5dc4ca02509b35a3f40b28",
    #     "VirusShare_85ba0aa794e2bab79947f2dc2a9fbed2",
    #     "VirusShare_15957ed1ff6bf19f2fa4c709409ebd70",
    #     "VirusShare_8e2b629a10625956f7609f2d939bcac4",
    #     "VirusShare_64191b00e77481347d55397bf033d597",
    #     "VirusShare_dada91fc9e8db0f7bd8a31d6f1ed75dc",
    #     "VirusShare_f06c6b9c3ccf73921f8570b10f9bb7fe",
    #     "VirusShare_d985eeb52e4407c39b53e7426a8bc2e3",
    #     "VirusShare_86539705c6eb59c6acaf55e580be653a"
    # ]
    # print(len(name))


    # name = [
    #     'VirusShare_97bc5adf5df9106efb885b78855c4838.gexf',  # 30.765587
    #     'VirusShare_e7ca640611fa2f8c630961199e13f6b5.gexf',  # 2.9279337
    #     'VirusShare_8e2b629a10625956f7609f2d939bcac4.gexf',  # 280.80453
    #     'VirusShare_63acec04855ac0c5641247f5ba3d48b9.gexf',  # 3.4467793
    #     'VirusShare_6ceede843c5dc4ca02509b35a3f40b28.gexf',  # 13.807023
    #     'VirusShare_ae165056c14a2cba5466cd69a28fc431.gexf',  # 6.2594376
    #     'VirusShare_86539705c6eb59c6acaf55e580be653a.gexf',  # 54.0898
    #     'VirusShare_b157472d61af978bee9d2c3b26df1e83.gexf',  # 15.3836565
    #     'VirusShare_5381c76ce28d84d3245efc4a19238d58.gexf',  # 6.5762854
    #     'VirusShare_8f7aff5ec7c3bb14331dfa3e981a0b73.gexf',  # 6.858757
    #     'VirusShare_6a3d9dfe6587141ace52c54d02e67e39.gexf',  # 6.5762854
    #     'VirusShare_6963b7ca41268cfa7470a3e8ad8e9766.gexf',  # 6.62138
    #     'VirusShare_41cec261cacf2f4bac3168740594361b.gexf',  # 0.7636745
    #     'VirusShare_d1df2a91bbe0594c1061ead71649ae09.gexf',  # 168.00703
    #     'VirusShare_74b1162820ca4095d8e911207a8a729a.gexf',  # 5.7805347
    #     'VirusShare_527c66dec303042a556a2349f29999dc.gexf',  # 6.621313
    #     'VirusShare_45e67a88d73488396eae77a7309d90e7.gexf',  # 6.621313
    #     'VirusShare_28a2725940ecb8b9686476cfa0dda209.gexf',  # 157.92831
    #     'VirusShare_c508d918a080b807fd23b51c350f86a5.gexf',  # 239.70909
    #     'VirusShare_9b0d37e5cb949ab86354788ce5488375.gexf',  # 69.2645
    #     'VirusShare_31e8119d2d0b14556eca26f06b679244.gexf',  # 6.5762854
    #     'VirusShare_32e2a5033829d8940bc50d5fb42f9785.gexf',  # 15.447115
    #     'VirusShare_d9263748ae4c6f2ec244351e054be5f4.gexf',  # 5.34118
    #     'VirusShare_250520d860f63afdf8c94affd0921253.gexf',  # 38.117565
    #     'VirusShare_0b2d190d17d50dfd4a589aca1a9caa49.gexf',  # 434.88388
    #     'VirusShare_fb8640f452beb0403462f3a036182ed7.gexf',  # 6.890682
    #     'VirusShare_f4db6fc9770ad3e0310f3e2a13b5cb8b.gexf',  # 6.5762854
    #     'VirusShare_6a9b2fac202d0ab867f89cd79c17f5c5.gexf',  # 6.5093713
    #     'VirusShare_17113d01df79f103c07f68b4ebfd51d3.gexf',  # 3.9143429
    #     'VirusShare_e3ae70d5e1f209bf4dc30ba013a165f2.gexf',  # 22.537872
    #     'VirusShare_0e78970a2de95e2d546c74cf9a134e75.gexf',  # 6.9511476
    #     'VirusShare_a0417e92830ad60ca9f58120a9af2f14.gexf',  # 5.000574
    #     'VirusShare_d985eeb52e4407c39b53e7426a8bc2e3.gexf',  # 142.4429
    #     'VirusShare_3b951e9452c02817a602753463958d67.gexf',  # 6.409724
    #     'VirusShare_15957ed1ff6bf19f2fa4c709409ebd70.gexf',  # 85.60583
    #     'VirusShare_5fe0c336f26b8eb2eb8c556299857cc1.gexf',  # 6.621313
    #     'VirusShare_962f12f1ef496da24b52eb87a8739521.gexf',  # 7.228757
    #     'VirusShare_0f54179597d9f1a5f075e7f64c722419.gexf',  # 15.447115
    #     'VirusShare_387636a84ffcf5f318599e9723375ad4.gexf',  # 6.5762854
    #     'VirusShare_038d756ffdb66cf297fb4bbc6cd994ba.gexf',  # 6.5762854
    #     'VirusShare_120712981dfad4dcf8bf085a74cc0baa.gexf',  # 6.858757
    #     'VirusShare_dd01bf58c4d54a5b5339b0b49e9799fb.gexf',  # 6.5762854
    #     'VirusShare_780a0e95e27fd60c516503fd9100e5bf.gexf',  # 6.858757
    #     'VirusShare_62f35131cf856d486a3433c9b94f8200.gexf',  # 193.92279
    #     'VirusShare_153bfed1fe7e7c813d9edb925764005f.gexf',  # 6.858757
    #     'VirusShare_bdc481637e36fccc0814df61cf8eb3d4.gexf',  # 6.5762854
    #     'VirusShare_7a35ec2f61888da33a78374ae73070e5.gexf',  # 4.1895137
    #     'VirusShare_2a944cca05ce869d504d2f1a15f66140.gexf',  # 382.2225
    #     'VirusShare_dbe9cb897c0a4b7edabb64a68c5a242f.gexf',  # 6.9511476
    #     'VirusShare_b93ea89c26caff768100b3b3734a9d74.gexf',  # 6.5762854
    #     'VirusShare_dbf2975d5765e960f6522a4b0128b81a.gexf',  # 10.421358
    #     'VirusShare_1cdebdc7075a2271ddc45c8fc19bda19.gexf',  # 1.8705304
    #     'VirusShare_ab0e4909e5316fa9886da9ec5f6bfa7c.gexf',  # 6.858757
    #     'VirusShare_0ecd899c27d8fb6e1c2ed5d7abeed74e.gexf',  # 63.353546
    #     'VirusShare_549130d50c2ce5a57debf51689c0a975.gexf',  # 6.858757
    #     'VirusShare_d33e77a93f902a7d5edf3e210539feeb.gexf',  # 6.5093713
    #     'VirusShare_9e68ea99c5b5bb5b17916f7c0c8191b7.gexf',  # 5.265791
    #     'VirusShare_4ae0387cd9f86182fe72e74afedbc4c1.gexf',  # 6.5762854
    #     'VirusShare_581fd33cc8acacb12cbbc715a766d9d9.gexf',  # 82.18351
    #     'VirusShare_8df785cbd4c179d4508d0090a90ef491.gexf',  # 6.2594376
    #     'VirusShare_2a16333a5cbe5f649977421336aded91.gexf',  # 15.457101
    #     'VirusShare_f06c6b9c3ccf73921f8570b10f9bb7fe.gexf',  # 169.74081
    #     'VirusShare_892e87a4d9955fe0d2e6e2b1edab635b.gexf',  # 6.1565742
    #     'VirusShare_2e814a5f5c114489ce38a117fabf3d18.gexf',  # 6.9511476
    #     'VirusShare_dada91fc9e8db0f7bd8a31d6f1ed75dc.gexf',  # 38.99535
    #     'VirusShare_6828f8c42b5a94471ca9faaddf9c1215.gexf',  # 3.103993
    #     'VirusShare_4fca330cfe3b6529e3d272305639533a.gexf',  # 6.198564
    #     'VirusShare_f2f9f752b08753c06fab05473a793c0a.gexf',  # 4.5687366
    #     'VirusShare_4adc49e9d85518866b4f95ab645faf6d.gexf',  # 6.2594376
    #     'VirusShare_69c2490f5478801eb72355368757d48a.gexf',  # 6.5762854
    #     'VirusShare_64191b00e77481347d55397bf033d597.gexf',  # 147.4903
    #     'VirusShare_4836ea8ec7adc537d17d1357caedf305.gexf',  # 86.3658
    #     'VirusShare_ff3e003183a3830d0eea2abd1220e602.gexf',  # 6.675957
    #     'VirusShare_d944797ebd43393a123a3c28f330fc5d.gexf',  # 3.103993
    #     'VirusShare_d59b8493b1ad4b600051c5c28d4b52b5.gexf',  # 353.60672
    #     'VirusShare_3539bdfad58ae71df12ddfe382bd6a80.gexf',  # 15.447115
    #     'VirusShare_1fb5801da767a22ea898bb90a0b94a5c.gexf',  # 15.460242
    #     'VirusShare_85ba0aa794e2bab79947f2dc2a9fbed2.gexf',  # 346.54233
    #     'VirusShare_26e37cae425e318cca08e9e46d84e113.gexf',  # 3.141282
    #     'VirusShare_3f3eb3a8b29bc9000c209701271e3184.gexf',  # 3.103993
    #     'VirusShare_91778619e3e81e0814ba2e69181dd6d4.gexf',  # 68.53075
    #     'VirusShare_160b455cc19e0c1ba1bef1b3a9ca97ce.gexf',  # 6.621313
    #     'VirusShare_3f5bc078b918e1a16050d8c545b8abbf.gexf',  # 3.103993
    #     'VirusShare_381f1339d5a0778606afd201759bc481.gexf',  # 6.5762854
    #     'VirusShare_f8c512a64f06173ba7b6f948b3cc3b81.gexf',  # 3.1992087
    #     'VirusShare_574e59fd51e7e894b296f684eaa37356.gexf',  # 334.88132
    #     'VirusShare_e0049eb5345cfe181d61a503ffbbc561.gexf',  # 15.3836565
    #     'VirusShare_0d88318b0186b5d65b87899dbaa7a5b3.gexf',  # 16.324898
    #     'VirusShare_f3ddd6fc9631a8679c1adefc3fa778b9.gexf',  # 434.88388
    #     'VirusShare_f07d5e4136ba20e54ba6ba422de971e4.gexf',  # 6.5762854
    #     'VirusShare_b4ca4cd07d5e87821012bdaa1faa2096.gexf',  # 329.44675
    #     'VirusShare_ef8c08d3f9c25c0c09a0e323d15df259.gexf',  # 3.103993
    #     'VirusShare_641b0063130668d6a25f5cd6bd8a7eb5.gexf',  # 6.62138
    #     'VirusShare_b5199fabcf3eb1baeb57d26ea2b733b2.gexf',  # 6.2594376
    #     'VirusShare_0a72229bb504d270e508b15ace1b38a4.gexf',  # 6.5762854
    #     'VirusShare_c2a1fb355e422a7fc515546af0886b66.gexf',  # 5.34118
    #     'VirusShare_2a33933c4cbfeb35f65ccbb9cb661866.gexf',  # 4.5069394
    #     'VirusShare_c2c60639213052cdda6872f345ea8e8f.gexf',  # 6.5762854
    #     'VirusShare_124347ab9424ccedbf7b841e007efc07.gexf',  # 6.5762854
    #     'VirusShare_1a25ab2e6585605665b282dea61a499d.gexf',  # 6.198564
    #     'VirusShare_97fec2ce6b34ac37a6a5df0739bd3860.gexf',  # 6.598372
    #     'VirusShare_00d0118a7152d850741d4143e968ba56.gexf',  # 6.858757
    #     'VirusShare_6f237d25472d9d09fc44ece7dc9ced92.gexf',  # 8.673773
    #     'VirusShare_cb2fa6dc53f32acad90a3cf4bc5d51f9.gexf',  # 7.057972
    #     'VirusShare_242e3e0fd9d9fecbe7f741a03c07c1de.gexf',  # 6.198564
    #     'VirusShare_836a62bec037576e17d16bb1bd036ffb.gexf',  # 176.4146
    #     'VirusShare_fbe403540869b62e2d3cc3acc639c074.gexf',  # 6.5762854
    #     'VirusShare_e95a8b7be2ce47237e8d1b808c93e8d3.gexf',  # 6.5762854
    #     'VirusShare_e5b3273d5f61c99dcd85328f9f3f34fd.gexf',  # 6.5762854
    #     'VirusShare_35565177740efd453fb60e63042d22eb.gexf',  # 4.766976
    #     'VirusShare_7f1d201c88fa16e39ea198fbc5b99553.gexf',  # 340.8121
    #     'VirusShare_bdad9ff85f4f1e00829d06db530f9eb1.gexf',  # 6.198564
    #     'VirusShare_e8c8a765a1ed3a746c3ac5c728e1202a.gexf',  # 6.5762854
    #     'VirusShare_8589ec219ffe9f94d16c07243bcb0631.gexf',  # 6.198564
    #     'VirusShare_4da7692874c056831c380935f8c04cc4.gexf',  # 15.990224
    #     'VirusShare_a7aec2cc8b5357d6dff9d21e94d623f2.gexf',  # 6.62138
    #     'VirusShare_9e6ccb1f074a1a68fb1bcc0436a76beb.gexf',  # 25.109325
    #     'VirusShare_0237bf35b128a6665f59d500b458ac0a.gexf',  # 6.198564
    #     'VirusShare_0e5512cffc5e9e51dd47450aa79434f5.gexf',  # 1.9641466
    #     'VirusShare_490934ef49d8ac537c69c2c537f9d17f.gexf',  # 2.9555335
    # ]
    # print("len(name)", len(name))

    # read attack samples from txt file
    # years = ['2018', '2019', '2020', '2021', '2022', '2023']
    # name = []
    # feature_dir = '/data/b/shiwensong/dataset/feature_Nov30/'
    # for year in years:
    #     txt_file = f'{feature_dir}/mar10_attack_samples_{year}_60.txt'
    #     with open(txt_file, 'r') as file:
    #         tmp = file.readlines()
    #     tmp = tmp[:20]
    #     name.extend(tmp)

    name = ['/data/b/shiwensong/dataset/virusshare2018_gexf/VirusShare_810bd45dc23b9694b1deb98b3b620ec4.gexf']

    print("selected_samples", len(name))
    print("selected_samples 0 ", name[0])

    all_combined_features = []
    # # 读取shap value
    with open('all_test_usx/RandomForest/120_samples_RandomForest_tree.pkl', 'rb') as f:
        shap_values = pickle.load(f)
    #
    shap_values = shap_values[1]
    print("shap_values", shap_values.shape)
    #
    combined_feature = None
    # cnt = 0
    #
    # find_name = set()
    dir = "/data/b/shiwensong/dataset/virusshare2018_gexf/"
    # name = name[0:10]
    for j in range(len(name)):
        print("j", j)
        shap_value = shap_values[j]
        file_name = name[j]
        print("file_name", file_name)
        fcg = FCG(file_name, 1)
        fcg.cal_centralities()
        degree = fcg.degree_feature
        katz = fcg.katz_feature
        closeness = fcg.closeness_feature
        harmonic = fcg.harmonic_feature
        combined_feature = np.hstack((degree, katz, closeness, harmonic))
        end_time = time.time()
        combined_feature = combined_feature.reshape(1, -1)
        print("combined_feature", combined_feature.shape)
        all_combined_features.append(combined_feature)
        res = attack_feature(shap_value, combined_feature, MLP_model, 0.95)
        print("res", res)
        if res is not None:
            cnt = cnt + 1
            # 把这个特征存储成pkl
            file_name = name[j].split('.')[0]
            with open('attack_feature/' + file_name + '.pkl', 'wb') as f:
                pickle.dump(res, f)
        else:
            print("failed")
            print("name", name[j])
    #
    # print("cnt", cnt)

    # all_combined_features = np.array(all_combined_features)
    # print("all_combined_features", all_combined_features.shape)
    # shap_values = get_shap(train_dataset, all_combined_features, MLP_model)
    # print("shap_values", shap_values[0].shape)
    # #存储shap value
    # with open('shap_values_28_samples_feb8.pkl', 'wb') as f:
    #     pickle.dump(shap_values, f)



    # #
    # #     mutation_list, new_fcg = random_m(fcg, 100)
    # #     new_fcg.cal_centralities()
    # #     new_degree = new_fcg.degree_feature
    # #     new_katz = new_fcg.katz_feature
    # #     new_closeness = new_fcg.closeness_feature
    # #     new_harmonic = new_fcg.harmonic_feature
    # #     new_combined_feature = np.hstack((new_degree, new_katz, new_closeness, new_harmonic))
    # #     all_combined_features.append(new_combined_feature)
    # #
    # #     mutation_list1, new_fcg1 = random_m(fcg, 500)
    # #     new_fcg1.cal_centralities()
    # #     new_degree1 = new_fcg1.degree_feature
    # #     new_katz1 = new_fcg1.katz_feature
    # #     new_closeness1 = new_fcg1.closeness_feature
    # #     new_harmonic1 = new_fcg1.harmonic_feature
    # #     new_combined_feature1 = np.hstack((new_degree1, new_katz1, new_closeness1, new_harmonic1))
    # #     all_combined_features.append(new_combined_feature1)
    # #
    # #     mutation_list2, new_fcg2 = random_m(fcg, 300)
    # #     new_fcg2.cal_centralities()
    # #     new_degree2 = new_fcg2.degree_feature
    # #     new_katz2 = new_fcg2.katz_feature
    # #     new_closeness2 = new_fcg2.closeness_feature
    # #     new_harmonic2 = new_fcg2.harmonic_feature
    # #     new_combined_feature2 = np.hstack((new_degree2, new_katz2, new_closeness2, new_harmonic2))
    # #     all_combined_features.append(new_combined_feature2)
    # #
    # # # 使用 np.vstack 堆叠所有的 combined_feature
    # stacked_combined_features = np.vstack(all_combined_features)
    # #
    # print("堆叠后的 combined_feature 形状:", stacked_combined_features.shape)
    # #
    # # #获取shap value
    # shap_values = get_shap(train_dataset, stacked_combined_features, MLP_model)
    # # #
    # # # #存储shap value
    # with open('shap_values_30_samples.pkl', 'wb') as f:
    #     pickle.dump(shap_values, f)


    #
    # # 是否有的点，即升，又降
    # shap_values = shap_values[0]
    # # cnt_1_degree = 0
    # # cnt_1_katz = 0
    # # cnt_1_closeness = 0
    # # cnt_1_harmonic = 0
    # # cnt_2_degree_katz = 0
    # # cnt_2_degree_closeness = 0
    # # cnt_2_degree_harmonic = 0
    # # cnt_2_katz_closeness = 0
    # # cnt_2_katz_harmonic = 0
    # # cnt_2_closeness_harmonic = 0
    # # cnt_3_degree_katz_closeness = 0
    # # cnt_3_degree_katz_harmonic = 0
    # # cnt_3_degree_closeness_harmonic = 0
    # # cnt_3_katz_closeness_harmonic = 0
    # cnt_4_degree_katz_closeness_harmonic = 0
    #
    #
    #
    # for i in range(40):
    #     if i % 4 == 3:
    #         print("i", i)
    #         analyze(i, shap_values[i-3], shap_values[i-2], shap_values[i-1], shap_values[i])
    #         print("===============")


        # degree = shap_values[i][:430]
        # katz = shap_values[i][430:860]
        # closeness = shap_values[i][860:1290]
        # harmonic = shap_values[i][1290:]
    #
    #     cnt_neg = 0
    #
    #     print("i", i)
    #     for j in range(430):
    #
    #         if degree[j] != 0 and katz[j] != 0 and closeness[j] != 0 and harmonic[j] != 0:
    #             # if degree[j] !=0 and katz[j] != 0 and closeness[j] != 0 and harmonic[j] != 0:
    #             cnt_4_degree_katz_closeness_harmonic += 1
    #             # elif degree[j] !=0 and katz[j] != 0 and closeness[j] != 0:
    #             #     cnt_3_degree_katz_closeness += 1
    #             # elif degree[j] !=0 and katz[j] != 0 and harmonic[j] != 0:
    #             #     cnt_3_degree_katz_harmonic += 1
    #             # elif degree[j] !=0 and closeness[j] != 0 and harmonic[j] != 0:
    #             #     cnt_3_degree_closeness_harmonic += 1
    #             # elif katz[j] != 0 and closeness[j] != 0 and harmonic[j] != 0:
    #             #     cnt_3_katz_closeness_harmonic += 1
    #             # elif degree[j] !=0 and katz[j] != 0:
    #             #     cnt_2_degree_katz += 1
    #             # elif degree[j] !=0 and closeness[j] != 0:
    #             #     cnt_2_degree_closeness += 1
    #             # elif degree[j] !=0 and harmonic[j] != 0:
    #             #     cnt_2_degree_harmonic += 1
    #             # elif katz[j] != 0 and closeness[j] != 0:
    #             #     cnt_2_katz_closeness += 1
    #             # elif katz[j] != 0 and harmonic[j] != 0:
    #             #     cnt_2_katz_harmonic += 1
    #             # elif closeness[j] != 0 and harmonic[j] != 0:
    #             #     cnt_2_closeness_harmonic += 1
    #             # elif degree[j] != 0:
    #             #     cnt_1_degree += 1
    #             # elif katz[j] != 0:
    #             #     cnt_1_katz += 1
    #             # elif closeness[j] != 0:
    #             #     cnt_1_closeness += 1
    #             # elif harmonic[j] != 0:
    #             #     cnt_1_harmonic += 1
    #
    #
    #             print("===============")
    #             print("j", j)
    #             print("positive")
    #
    #             print("degree", degree[j])
    #             print("katz", katz[j])
    #             print("closeness", closeness[j])
    #             print("harmonic", harmonic[j])
    #             print("===============")
    #     # print("cnt_1_degree", cnt_1_degree)
    #     # print("cnt_1_katz", cnt_1_katz)
    #     # print("cnt_1_closeness", cnt_1_closeness)
    #     # print("cnt_1_harmonic", cnt_1_harmonic)
    #     # print("cnt_2_degree_katz", cnt_2_degree_katz)
    #     # print("cnt_2_degree_closeness", cnt_2_degree_closeness)
    #     # print("cnt_2_degree_harmonic", cnt_2_degree_harmonic)
    #     # print("cnt_2_katz_closeness", cnt_2_katz_closeness)
    #     # print("cnt_2_katz_harmonic", cnt_2_katz_harmonic)
    #     # print("cnt_2_closeness_harmonic", cnt_2_closeness_harmonic)
    #     # print("cnt_3_degree_katz_closeness", cnt_3_degree_katz_closeness)
    #     # print("cnt_3_degree_katz_harmonic", cnt_3_degree_katz_harmonic)
    #     # print("cnt_3_degree_closeness_harmonic", cnt_3_degree_closeness_harmonic)
    #     # print("cnt_3_katz_closeness_harmonic", cnt_3_katz_closeness_harmonic)
    #     print("cnt_4_degree_katz_closeness_harmonic", cnt_4_degree_katz_closeness_harmonic)








        #get the shap value of the sample
        # shap_values = get_shap(train_dataset, combined_feature, MLP_model)
        # print(len(shap_values))
        # print("shap_values", shap_values)
        # #shape[0]
        # class_0 = shap_values[0]
        #
        # #读取txt文件
        # important_sensitive_apis = []
        # with open('important_sensitive_apis.txt', 'r') as f:
        #     for line in f.readlines():
        #         important_sensitive_apis.append(line.strip())
        #
        # print(len(important_sensitive_apis))
        #
        # cnt_0 = 0
        # cnt_pos = []
        # cnt_neg = []
        # cnt_430_pos = set()
        # cnt_430_neg = set()
        # new_feature = []
        # # print("class_0", class_0.shape) #（1，1720）
        # for i in range(1720):
        #     # print("i", i)
        #     # print("important_sensitive_apis", important_sensitive_apis[i])
        #     value = class_0[i]
        #     old_feature = combined_feature[0][i]
        #     if value > 0:
        #         cnt_pos.append(i)
        #         print("positive")
        #         print("i", i)
        #         print("old_feature", old_feature)
        #         print("shap", value)
        #         cnt_430_pos.add(i%430)
        #     elif value < 0:
        #         cnt_neg.append(i)
        #         cnt_430_neg.add(i%430)
        #         print("negative")
        #         print("i", i)
        #         print("old_feature", old_feature)
        #         print("shap", value)
        #     else:
        #         cnt_0 += 1
        #     # print("value", value)
        #
        # #对cnt_pos和cnt_neg进行下标排序
        # print("cnt_pos", cnt_pos)
        # print("cnt_neg", cnt_neg)
        # idx_pos = np.argsort(cnt_pos)[::-1]
        # idx_neg = np.argsort(cnt_neg)
        #
        # #对于正值
        # for i in range(len(cnt_pos)):
        #     index = cnt_pos[i]
        #     target = combined_feature[0][index]
        #     if target == 0:
        #         continue
        #     else:
        #         modify = 0.1
        #         combined_feature[0][index] = target - cnt_pos[i] * modify
        #         pros = MLP_model.predict(combined_feature)
        #         print("pros", pros)
        #         # if pros[0][0] < 0.5:
        #         #     print("success")
        #         #     break
        #
        # print("======", len(cnt_pos))
        #
        #
        # for i in range(len(cnt_neg)):
        #     index = cnt_neg[i]
        #     target = combined_feature[0][index]
        #     if target == 0:
        #         continue
        #     else:
        #         modify = 0.1
        #         # if target - modify > 0:
        #         combined_feature[0][index] = target + cnt_neg[i] * modify
        #         pros = MLP_model.predict(combined_feature)
        #         print("pros", pros)
        #         # if pros[0][0] < 0.5:
        #         #     break
        #
        #
        #
        # print("cnt_pos", len(cnt_pos))
        # print("cnt_neg", len(cnt_neg))
        # print("cnt_430_pos", len(cnt_430_pos))
        # print("cnt_430_neg", len(cnt_430_neg))
        #
        # intersection = cnt_430_pos.intersection(cnt_430_neg)
        # print(intersection)
        #
        # print("cnt_0", cnt_0)
        #
        # print("==========finish==========")


