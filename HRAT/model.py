import os
#设置GPU编号
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#reinforcement learning modify CFG
#environment: CFGModifierEnvConstraints, and return the reward

# 这段代码实现了一个基于深度Q网络（DQN）的强化学习模型，用于学习如何在一个特定环境中做出最优决策。主要组成部分包括：
#
# 网络结构（Net类）：定义了一个神经网络，包括输入层、隐藏层和输出层。这个网络用于近似Q函数，预测每个动作的价值。
#
# DQN类：
#
# choose_action：根据当前状态选择动作。它使用eval_net网络来预测每个动作的价值，并根据这些预测选择动作。
# store_transition：在内存中存储状态转换，包括当前状态、动作、奖励和下一个状态。
# learn：从内存中随机抽取一批样本来训练网络。这里使用目标网络（target_net）来计算目标Q值，并用评估网络（eval_net）来计算预测的Q值，然后通过最小化这两个值之间的差异来更新网络权重。
# 这个DQN模型通过交互学习环境来进行学习，每次交互后，它会将经验（状态、动作、奖励、新状态）存储在内存中。在训练阶段，它会从内存中随机抽取一些经验来更新网络，使其更好地估计在不同状态下采取不同动作的期望回报。通过这种方式，模型逐渐学会如何在给定环境中做出最优决策。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:  "+str(device))
# device = 'cpu'
class Net(nn.Module):
    def __init__(self, states_dim, actions_num):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(states_dim, 1000)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(1000, 50)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, actions_num)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        self.act = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        actions_value = self.out(x)
        return self.softmax(actions_value)


class DQN(object):
    """
    Description:

    Source:
        Self designed. This environments is designed to modify function call graph in programming.

    Observations:
        Type: vector:
            the centrality of sensitive apis (attention nodes)

    Actions:
        Type: list(4)
        Num     Action
        0       add an edge between two nodes
        1       rewiring
        2       add an nodes, and connect it to another nodes
        3       delete an nodes


    Reward:
        Reward is


    Episode Termination:
        The observation is classified as target class.
        Episode length is greater than 200.

    """

    def __init__(self, states_dim, actions_num, memory_capacity, learning_rate, ):

        self.eval_net, self.target_net = Net(states_dim, actions_num).to(device), Net(states_dim, actions_num).to(device)
        self.learn_step_counter = 0
        # self.action_space = ['add_edge', 'rewiring', 'add_nodes', 'delete_nodes']
        
        # For recovery
        # self.action_space = ['add_edge', 'rewiring', 'delete_nodes'] 

        self.memory_counter = 0  # for storing memory
        # modified by ssw , discard the numpy and instead of torch
        # self.memory = np.zeros((memory_capacity, states_dim * 2 + 5))  # initialize memory
        self.memory = torch.zeros((memory_capacity, states_dim * 2 + 5)).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, actions_num, EPSILON=0.8):
        # x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = torch.unsqueeze(x.float(), 0).to(device)
        # input only one sample
        if np.random.uniform() < EPSILON:
            action_type = self.eval_net.forward(x)
            # print("predicted action type:   "+str(action_type))
            # action_type = torch.max(action_type, 1)[1].data.cpu().numpy()
            action_type = torch.max(action_type, 1)[1]
            action_type = action_type.to(device)  # 将选择的动作移动到GPU
        else:  # random
            # action = np.random.randint(actions_num)
            # action_type = action
            action_type = torch.tensor(np.random.randint(actions_num), dtype=torch.float32).to(device)

        return action_type

    def store_transition(self, s, a, r, s_, memory_capacity):
        # modified by ssw s, r is a tensor
        # ensure the data type is torch
        # r = np.array(r)
        # transition = np.hstack((s.cpu(), a, r, s_.cpu())) # hstack: horizontal stack
        # 转成二维，shape is (40000多, 1)
        # a = a.view(-1, 1)
        # s = s.view(-1, 1)
        # s_ = s_.view(-1, 1)
        r = r.unsqueeze(0)  # Convert r to a 1D tensor
        transition = torch.cat((s, a, r, s_), dim=0)
        # print("transition:   "+str(transition))
        # replace the old memory with new memory
        index = self.memory_counter % memory_capacity
        # modified by ssw reshape
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, MEMORY_CAPACITY, BATCH_SIZE, N_STATES, TARGET_REPLACE_ITER=10, GAMMA=0.9):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # b_state = torch.FloatTensor(b_memory[:, :N_STATES]).to(device) # N_States : state dimension = degree dimension
        # b_action = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)).to(device) # action dimension: 4 []
        b_state = torch.tensor(b_memory[:, :N_STATES], dtype=torch.float32).to(device)
        b_action = torch.tensor(b_memory[:, N_STATES:N_STATES + 1], dtype=torch.long).to(device)
        # b_reward = torch.FloatTensor(b_memory[:, N_STATES + 4:N_STATES + 5]).to(device)
        b_reward = torch.tensor(b_memory[:, N_STATES + 4:N_STATES + 5], dtype=torch.float32).to(device)
        # b_state_new = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)
        b_state_new = torch.tensor(b_memory[:, -N_STATES:], dtype=torch.float32).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_state).gather(1, b_action)  # shape (batch, 1)
        q_next = self.target_net(b_state_new).detach()  # detach from graph, don't backpropagate
        q_target = b_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # print("current loss is: " + str(loss))


        self.optimizer.zero_grad()
        loss.backward()

        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1)

        self.optimizer.step()
