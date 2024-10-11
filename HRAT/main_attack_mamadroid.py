import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pickle
import numpy as np
from tqdm import tqdm
import torch

from Utils import degree_centrality_torch, katz_feature_torch, to_adjmatrix, check_folder,\
    find_nn_torch, trans2triple_rw
import myenv_withconstraints_mamadroid
from Utils import parse_arguments
import os
import time
import joblib

from model import DQN

m = 100
steep = 1
MEMORY_CAPACITY = 100
actions_num = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = 'cuda:2'

if __name__ == '__main__':

    save_path = "results_120"
    check_folder(save_path)

    ## load train data
    # train_path = "data_train.pkl"# pkl is serialized by pickle
    # train_path = "new_data_train_2023.pkl"
    # train_path = "new_data_train_201820222023.pkl"
    train_path = "/data/b/shiwensong/dataset/feature_Nov30/mamadroid_family.pkl"
    df = open(train_path, "rb")
    train_feature,train_label = pickle.load(df) #format: dict
    # print('train--length',len(data_train_dict)) # length 1883
    df.close()

    
    print(train_feature.shape)
    print(type(train_feature))
    print(train_label.shape)
    print(type(train_label))
    # 打印前几行数据
    # num_rows_to_print = 5  # 例如，打印前5行
    #
    # print("\nFirst", num_rows_to_print, "sha256 values:")
    # print(train_sha256[:num_rows_to_print])
    #
    # print("\nFirst", num_rows_to_print, "features:")
    # print(train_feature[:num_rows_to_print])
    #
    # print("\nFirst", num_rows_to_print, "labels:")
    # print(train_label[:num_rows_to_print])

    # load test
    test_data = "/data/b/shiwensong/dataset/testset_2022_430features_mamadroid.pkl"
    # test_data = 'data_test.pkl'
    # test_data = "/data/b/shiwensong/dataset/testset_118.pkl"
    df = open(test_data, "rb")
    data_test_dict = pickle.load(df)
    df.close()
    print('test--length', len(data_test_dict))
    test_sha256_list = list(data_test_dict)
    print("==== loading test adjacent matrix ====")
    print(list(data_test_dict.values())[:1])
    test_adj = [tmp["adjacent_matrix"] for tmp in tqdm(data_test_dict.values())]
    print("==== loading test sensitive api index ====")
    test_mamadroid_matrix = [tmp["mamadroid_matrix"] for tmp in tqdm(data_test_dict.values())]
    print(len(test_sha256_list))
    test_mamadroid_dict = [tmp["mamadroid_dict"] for tmp in tqdm(data_test_dict.values())]
    print(len(test_mamadroid_dict))


    seq_save_path = save_path + "/mamadroid_actionseq_2022_430features_memory100"
    if not os.path.exists(seq_save_path):
        os.mkdir(seq_save_path)

    print("current device: ", torch.cuda.get_device_name(device))
    # 获取稀疏矩阵的非零元素的行索引、列索引和值
    row, col = train_feature.nonzero()
    data = train_feature.data

    # 创建 PyTorch 的稀疏张量
    shape = train_feature.shape
    train_feature_torch = torch.sparse.FloatTensor(torch.LongTensor([row, col]), torch.FloatTensor(data),
                                                   torch.Size(shape)).to(device)
    train_label_torch = torch.from_numpy(np.array(train_label)).to(device)

    dqn = DQN(states_dim=train_feature_torch.shape[1],
                    actions_num=actions_num,
                    memory_capacity=MEMORY_CAPACITY,
                    learning_rate=0.01)

    cnt = 0

    for zidx in tqdm(range(0, len(test_sha256_list))):
        print("\ncurrent test number:\t " + str(zidx))

        X_test_sha256 = test_sha256_list[zidx]
        print(X_test_sha256)

        begin_time = time.time()
        
        print("==== transfer adjacent matrix to triple set ====")
        X_test_am = test_adj[zidx]
        # modified by ssw， 太大跳过
        print('shape',X_test_am.shape)
        if X_test_am.shape[0] > 12000:
            print('large graph')
            large = seq_save_path + "/large.txt"
            file_time = open(large, 'a')
            ans = "sha256: " + X_test_sha256 + " X_test_triple.shape[0]:" + str(X_test_am.shape[0]) + "\n"
            file_time.write(ans)
            file_time.close()
            continue

        X_test_mamadroid_matrix = test_mamadroid_matrix[zidx]
        X_test_mamadroid_dict = test_mamadroid_dict[zidx]
        print(X_test_mamadroid_matrix.shape)
        triple_path = save_path + "/triple_set"
        check_folder(triple_path)
        if not os.path.exists(triple_path):
            os.mkdir(triple_path)
        X_test_triple = trans2triple_rw(X_test_am, X_test_sha256, triple_path, False)
        if X_test_triple is None:
            none = seq_save_path + "/none.txt"
            file_time = open(none, 'a')
            ans = "sha256: " + X_test_sha256 + "\n"
            file_time.write(ans)
            file_time.close()
            continue
        print("finish")

        node_number = X_test_am.shape[0]
        adj_torch = to_adjmatrix(X_test_triple, node_number)
        adj_torch = adj_torch.to(device)
        # degree_fea = degree_centrality_torch(adj_torch, X_test_sen_idx, device)
        # katz_fea = katz_feature_torch(adj_torch, X_test_sen_idx, device=device)
        # katz_fea = torch.squeeze(katz_fea)
        # # close_fea = closeness_centrality_torch(adj_torch, X_test_sen_idx, device=device)
        # # harm_fea = harmonic_centrality_torch(adj_torch, X_test_sen_idx, device=device)
        # # X_test_feature = torch.cat((degree_fea, torch.squeeze(katz_fea), torch.squeeze(close_fea), torch.squeeze(harm_fea)), 0) #特征？
        #
        # X_test_feature = torch.cat((degree_fea, torch.squeeze(katz_fea)), 0)  # 特征？


        #单特征的测试
        # X_test_feature = degree_fea
        # cnt = 0
        # for i in range(len(X_test_sen_idx)):
        #     if X_test_sen_idx[i] != -1:
        #         cnt += 1
        # print('sensitive api number:', cnt)
        # non_zero = torch.nonzero(X_test_feature)
        # print('non_zero',non_zero)
        ##
        # y, min_dist = find_nn_torch(X_test_feature, train_feature_torch, train_label_torch, k=1)
        # print('y',y)
        #
        # if y == 0:
        #     print('==== data cannot be correctly classified as malware ====\t')
        #     error = seq_save_path + "/error.txt"
        #     file_time = open(error, 'a')
        #     ans = "sha256: " + X_test_sha256 + " X_test_triple.shape[0]:" + str(X_test_triple.shape[0]) + "\n"
        #     file_time.write(ans)
        #     file_time.close()
        #     cnt += 1
        #     continue
        y_test = torch.tensor(1).to(device)
        # begin train something
        print('\t ==== get the nearest neighbors for optimization ====')

        w = (2 * (train_label_torch != y_test).int() - 1).float().to(device)

        # load constraints
        print('\t ===== loading constraints =====')
        # constraints_path = "constraints/" + X_test_sha256 + ".txt"
        constraints_path = "/data/b/shiwensong/dataset/120_samples_constraints/cons/" + X_test_sha256 + ".txt"
        tmp = open(constraints_path, "r", encoding='utf-8').readlines()
        constraints = [int(a.replace("\n", "")) for a in tmp]
        # modified by ssw
        # constraints = np.array(constraints)
        constraints = torch.tensor(constraints, dtype=torch.int8).to(device)

        # modified by ssw transfer to torch
        X_test_triple = torch.from_numpy(X_test_triple).float().to(device)
        X_test_mamadroid_matrix = torch.from_numpy(X_test_mamadroid_matrix).to(device)
        tensor_list = []

        # 遍历 NumPy 数组中的每个元素，每个元素是一个键值对的元组
        for key, value_set in X_test_mamadroid_dict.items():
            value_list = list(value_set)
            value_tensor = torch.tensor(value_list, dtype=torch.float32).to(device)
            # 将键和张量作为一个元组添加到列表中
            tensor_list.append((key, value_tensor))

        # 将列表转换为字典
        X_test_mamadroid_dict = dict(tensor_list)
        print('X_test_mamadroid_dict', type(X_test_mamadroid_dict))
        # X_test_sen_idx = torch.from_numpy(X_test_sen_idx).to(device)
        # print('original label', y_test)
        # print('X_test_triple device', X_test_triple.device)
        # print('y_test label', y_test.device)
        # print('X_test_sen_idx device', X_test_sen_idx.device)
        # print('train_feature_torch device', train_feature_torch.device)
        # print('train_label_torch device', train_label_torch.device)
        # print('w device', w.device)
        # print('constraints device', constraints.device)
        # print('adj_torch device', adj_torch.device)

        steep = torch.tensor(steep).to(device)
        knn1 = joblib.load(
            '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_1000samples/familyknn_1.pkl')
        env = myenv_withconstraints_mamadroid.CFGModifierEnvConstraints(target_graph=X_test_triple,
                                                              label=y_test,
                                                              target_mamadoir_dict=X_test_mamadroid_dict,
                                                              node_num=node_number,
                                                              X_train=train_feature_torch,
                                                              y_train=train_label_torch,
                                                              train_set=train_feature_torch,
                                                              train_label=train_label_torch,
                                                              w=w, steep=steep,
                                                              constraints=constraints,
                                                                model = knn1)

        tmp_graph = adj_torch.float()
        mamadroid_feature = env.getMamadroidFeature(tmp_graph, X_test_mamadroid_dict)
        print('mamadroid',mamadroid_feature.shape)
        print('X_test_mamadroid_dict', X_test_mamadroid_dict.keys())
        #flatten
        mamadroid_feature = mamadroid_feature.flatten().reshape(1,-1)
        print('mamadroid',mamadroid_feature.shape)
        
        
        # degree = env.getDegreeCentrality(X_test_triple, X_test_sen_idx)
        # cnt = 0
        # for i in range(len(X_test_sen_idx)):
        #     if X_test_sen_idx[i] != -1:
        #         cnt += 1
        # print('sensitive api number:', cnt)
        # katz = env.katz_feature_torch(adj_torch, X_test_sen_idx)
        # print("adj_torch device", adj_torch.device)

        # print("tmp device", tmp.device)
        # closeness = env.getClosenessCentrality(tmp_graph, X_test_sen_idx)
        non_zero = torch.nonzero(mamadroid_feature)
        print('non_zero', non_zero.size())

        # cur_label, min_dist = env.getlabel(mamadroid_feature)
        # 转成numpy
        mamadroid_feature = mamadroid_feature.cpu().numpy()
        cur_label = knn1.predict(mamadroid_feature)
        cur_label = cur_label[0]
        print('cur_label', cur_label)
        if cur_label == 0:
            print('==== data cannot be correctly classified as malware ====\t')
            error = seq_save_path + "/error.txt"
            file_time = open(error, 'a')
            ans = "sha256: " + X_test_sha256 + " X_test_triple.shape[0]:" + str(X_test_triple.shape[0]) + "\n"
            file_time.write(ans)
            file_time.close()
            cnt += 1
            continue


        print(train_feature_torch.shape)
        print(train_label_torch.shape)
        print(X_test_triple.shape)

        print('\t ==== Collecting experience ... ====')
        flag = 0
        for i_episode in range(30):
            actions_store = []
            print("reset")
            state = env.reset()
            # modified by ssw
            # ep_r = 0
            ep_r = torch.tensor(0.0).to(device)
            count = 0

            # modified by ssw
            is_done = False

            while True:
                print("current round:\t" + str(count) + "\t==========")
                if dqn.memory_counter <= MEMORY_CAPACITY:
                    # action_type = np.random.randint(actions_num)
                    action_type = torch.tensor(np.random.randint(actions_num), dtype=torch.float16).to(device)
                    # action_type = torch.tensor(3.0)
                    # print("random action type1 :   "+str(action_type))
                else:
                    # action_type = dqn.choose_action(state, actions_num=actions_num)
                    action_type = dqn.choose_action(state, actions_num=actions_num)
                    # print("random action type2 :   " + str(action_type))


                state_, reward, done, info, cur_graph = env.step(action=action_type)
                # print('reward：'+str(reward))
                # if type(action_type) is np.ndarray:
                #     action_type = action_type[0]

                if isinstance(action_type, torch.Tensor):
                    action_type = action_type.item()  # 提取标量值
                    action_type = torch.tensor(action_type, dtype=torch.float16).to(device)

                # action = np.array([action_type] + info)
                # 在这里，action_type已经是GPU上的张量，info也是在GPU上的张量
                # print("info:   "+str(info))
                info_tensor = [torch.tensor(item, dtype=torch.float16).to(device) for item in info]
                # print("action type:   " + str(action_type))
                action = [action_type] + info_tensor
                action = torch.stack(action, dim=0)
                # print("action:   "+str(action))
                dqn.store_transition(state, action, reward, state_, MEMORY_CAPACITY)
                # print("\t", action.tolist())
                actions_store.append(action.tolist())
                ep_r += reward

                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn(MEMORY_CAPACITY, 8, N_STATES=state.shape[0])

                    if done:
                        tmp = ep_r.item()
                        print("Ep:", i_episode,
                            ' | Ep_reward: ', round(tmp, 2))
                if done:
                    if count<5:
                        flag = 1
                    # 220310 for check : can be commented
                    check_label,min_dist = find_nn_torch(state_, train_feature_torch, train_label_torch, k=1)
                    if check_label == 1:
                        print("something went wrong")

                    action_path = seq_save_path + "/" + X_test_sha256 + "action_list" + str(i_episode) + ".txt"

                    file = open(action_path, 'w')
                    for az in actions_store:
                        file.write(str(az))
                        file.write('\n')
                    file.write(str(done))
                    file.write('\n')
                    file.close()

                    graph_path = seq_save_path + "/final_graph" + str(i_episode)+"_" + X_test_sha256 + ".npy"
                    # modified by ssw
                    # np.save(graph_path, cur_graph)
                    torch.save(cur_graph, graph_path)

                    feature_file_name = seq_save_path + "/final_feature_epi" + str(i_episode)+"_" + X_test_sha256 + ".txt"
                    file_feature = open(feature_file_name, 'w')
                    file_feature.write(str(state_.tolist()))
                    file_feature.write('\n')
                    file_feature.write(str(state.tolist()))
                    file_feature.write('\n')
                    file_feature.close()
                    print('!!!! finish within 5')

                    is_done = True

                    end_time = time.time()
                    #写入文件
                    finish_time_count = seq_save_path + "/time.txt"
                    file_time = open(finish_time_count, 'a')
                    ans = "sha256: " + X_test_sha256 + " episode: "+ str(i_episode) + " count: " + str(count) + " finished_time: " + str(end_time-begin_time) + " reward: " + str(ep_r.item()) + " X_test_triple.shape[0]:" + str(X_test_triple.shape[0]) + "\n"
                    file_time.write(ans)
                    file_time.close()


                    break
                if count > 500:
                    break
                s = state_
                count += 1
            if flag == 1:
                break
            if is_done:
                break

            #memory management
            torch.cuda.empty_cache()
            # 删除不再需要的变量
            del state, state_, action, cur_graph

            end_time = time.time()
            # 写入文件
            finish_time_count = seq_save_path + "/time.txt"
            file_time = open(finish_time_count, 'a')
            ans = "sha256: " + X_test_sha256 + " episode: " + str(i_episode) + " count: " + str(
                count) + " finished_time: " + str(end_time - begin_time) + " reward: " + str(
                ep_r.item()) + " X_test_triple.shape[0]:" + str(X_test_triple.shape[0]) + "\n"
            file_time.write(ans)
            file_time.close()
    print('cnt',cnt)