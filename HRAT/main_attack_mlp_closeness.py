import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pickle
import numpy as np
from tqdm import tqdm
import torch

from Utils import degree_centrality_torch, katz_feature_torch, to_adjmatrix, check_folder,\
    find_nn_torch, trans2triple_rw
import myenv_withconstraints_mlp_closeness
from Utils import parse_arguments
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy

from model import DQN

m = 100
steep = 1
MEMORY_CAPACITY = 100
actions_num = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = 'cuda:2'

class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 最后一层不使用激活函数

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 直接返回 logits
        return x

if __name__ == '__main__':

    args = parse_arguments()

    save_path = args.save_dir
    check_folder(save_path)

    MLP_model = MLPModel(input_size=21986)
    model_path = args.attack_mlp
    MLP_model.load_state_dict(torch.load(model_path))
    MLP_model.to(device)
    MLP_model.eval()


    train_path = args.train_set
    df = open(train_path, "rb")
    data_train_dict = pickle.load(df)
    df.close()


    train_sha256 = data_train_dict["sha256"]
    train_feature = data_train_dict["feature"]
    train_label = data_train_dict["label"]



    # load test
    test_data = args.test_set
    df = open(test_data, "rb")
    data_test_dict = pickle.load(df)
    df.close()
    print('test--length', len(data_test_dict))
    test_sha256_list = list(data_test_dict)
    print("==== loading test adjacent matrix ====")
    print(list(data_test_dict.values())[:5])
    test_adj = [tmp["adjacent_matrix"] for tmp in tqdm(data_test_dict.values())]
    print("==== loading test sensitive api index ====")
    test_sensi_idx = [tmp["sensitive_api_list"] for tmp in tqdm(data_test_dict.values())]
    print(len(test_sha256_list))

    seq_save_path = save_path + "/closeness_actionseq_mlp"
    if not os.path.exists(seq_save_path):
        os.mkdir(seq_save_path)

    print("current device: ", torch.cuda.get_device_name(device))
    train_feature_torch = torch.from_numpy(np.array(train_feature)).to(device)
    train_label_torch = torch.from_numpy(np.array(train_label)).to(device)


    dqn = DQN(states_dim=train_feature_torch.shape[1],
                    actions_num=actions_num,
                    memory_capacity=MEMORY_CAPACITY,
                    learning_rate=0.01)

    cnt = 0

    for zidx in tqdm(range(0, len(test_sha256_list))):
        try:
            print("\ncurrent test number:\t " + str(zidx))

            X_test_sha256 = test_sha256_list[zidx]
            print(X_test_sha256)

            begin_time = time.time()

            print("==== transfer adjacent matrix to triple set ====")
            X_test_am = test_adj[zidx]
            # modified by ssw， 太大跳过
            # print('shape',X_test_am.shape)
            if X_test_am.shape[0] > 12000:
                print('large graph')
                large = seq_save_path + "/large.txt"
                file_time = open(large, 'a')
                ans = "sha256: " + X_test_sha256 + " X_test_triple.shape[0]:" + str(X_test_am.shape[0]) + "\n"
                file_time.write(ans)
                file_time.close()
                continue

            X_test_sen_idx = test_sensi_idx[zidx]
            # print(X_test_sen_idx.shape)
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
            # adj_torch = to_adjmatrix(X_test_triple, node_number)
            # print('adj_torch', adj_torch)
            # degree_fea = degree_centrality_torch(adj_torch, X_test_sen_idx, device)
            # katz_fea = katz_feature_torch(adj_torch, X_test_sen_idx, device=device)
            # non_zero_degree = torch.nonzero(degree_fea)
            # non_zero_katz = torch.nonzero(katz_fea)
            # print('non_zero_degree:', non_zero_degree)
            # print('non_zero_katz:', non_zero_katz)

            # katz_fea = torch.squeeze(katz_fea)
            # # close_fea = closeness_centrality_torch(adj_torch, X_test_sen_idx, device=device)
            # # harm_fea = harmonic_centrality_torch(adj_torch, X_test_sen_idx, device=device)
            # # X_test_feature = torch.cat((degree_fea, torch.squeeze(katz_fea), torch.squeeze(close_fea), torch.squeeze(harm_fea)), 0) #特征？
            #
            # X_test_feature = torch.cat((degree_fea, torch.squeeze(katz_fea)), 0)  # 特征？


            #单特征的测试
            # X_test_feature = degree_fea
            ##
            # y, min_dist = find_nn_torch(X_test_feature, train_feature_torch, train_label_torch, k=1)
            # score = MLP_model(X_test_feature)
            # print('score:', score)
            # if score < 0:
            #     y = 0
            # else:
            #     y = 1


            y_test = 1
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
            X_test_sen_idx = torch.from_numpy(X_test_sen_idx).to(device)
            print('original label', y_test)
            env = myenv_withconstraints_mlp_closeness.CFGModifierEnvConstraints(target_graph=X_test_triple,
                                                                                label=y_test,
                                                                                target_sen_api_idx=X_test_sen_idx,
                                                                                node_num=node_number,
                                                                                w=w, steep=steep,
                                                                                constraints=constraints, mlp_model=MLP_model)

            # feature = env.getDegreeCentrality(X_test_triple, X_test_sen_idx)
            adj_torch = to_adjmatrix(X_test_triple, node_number)
            adj_torch = adj_torch.to(device)
            # feature = env._feature_torch(adj_torch, X_test_sen_idx)
            feature = env.getClosenessCentrality(adj_torch, X_test_sen_idx)
            print('feature:')
            print(torch.nonzero(feature).size())
            feature = torch.squeeze(feature)
            score = MLP_model(feature)
            print('score:', score.item())
            if score < 0:
                print('==== data cannot be correctly classified as malware ====\t')
                error = seq_save_path + "/error.txt"
                file_time = open(error, 'a')
                ans = "sha256: " + X_test_sha256 + " X_test_triple.shape[0]:" + str(X_test_triple.shape[0]) + "\n"
                file_time.write(ans)
                file_time.close()
                cnt += 1
                continue


            # print(train_feature_torch.shape)
            # print(train_label_torch.shape)
            # print(X_test_triple.shape)

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
                        # actions_num_scalar = actions_num.item()
                        # action_type = torch.randint(0, actions_num_scalar, (1,), dtype=torch.float16, device=device)
                    # action_type = torch.tensor(3.0)
                    #     print("random action type1 :   "+str(action_type))
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
                        end_time = time.time()
                        # 写入文件
                        finish_time_count = seq_save_path + "/time.txt"
                        file_time = open(finish_time_count, 'a')
                        ans = "sha256: " + X_test_sha256 + " episode: " + str(i_episode) + " count: " + str(
                            count) + " finished_time: " + str(end_time - begin_time) + " reward: " + str(
                            ep_r.item()) + " X_test_triple.shape[0]:" + str(X_test_triple.shape[0]) + "\n"
                        file_time.write(ans)
                        file_time.close()

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
        except Exception as e:
            print(f"An error occurred: {e}")

    print('cnt',cnt)

