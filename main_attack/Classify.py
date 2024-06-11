import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from itertools import islice
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 为每个 GPU 设置内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
import sys
import os
sys.path.append(os.path.abspath('../type'))
sys.path.append(os.path.abspath('task'))
import itertools


from ExtractFeature import obtain_sensitive_apis
from concurrent.futures import ThreadPoolExecutor


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import argparse
import csv
from sklearn.tree import DecisionTreeClassifier  # 或 DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier  # 或 RandomForestRegressor
import matplotlib.pyplot as plt
import shap
import xgboost
from tensorflow.keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard
import time
import numpy as np
import os
import tensorflow as tf
import keras.backend as K

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch_geometric
from torch_geometric.utils import from_networkx
import torch.optim as optim
import pandas as pd
import random
from multiprocessing import Pool as ThreadPool
from functools import partial
import networkx as nx
import pandas as pd
from logging import Logger
import pickle
from scipy.sparse import vstack, csr_matrix
from sklearn.decomposition import PCA
from joblib import dump, load

from FCG import FCG
from mlp import MLP, mlp_train, mlp_evaluate
from data import SimpleDataset

from sklearn.linear_model import PassiveAggressiveClassifier
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Lambda, Flatten, Reshape
from setting import init_logger
from gnn_pyq import GNN
from FCG import FCG
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

save_path = './430features_5yearsdataset_all/'


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

def train_mlp_model(vector, label, input_size, model_output_path, type, epochs=100, batch_size=16):
    # 初始化模型
    model = MLPModel(input_size=input_size)
    model.train()

    # 使用 nn.BCEWithLogitsLoss 作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCEWithLogitsLoss()

    # 准备数据加载器
    dataset = torch.utils.data.TensorDataset(vector, label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target.view(-1, 1).float())  # 注意: target 应当是 float 类型
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # 保存模型
    torch.save(model.state_dict(), model_output_path + type + 'MLP.pth')
    return model_output_path + type + 'MLP.pth'

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(latent_dim, input_shape):
    # 编码器
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(512, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # 解码器
    decoder_h = Dense(512, activation='relu')
    decoder_mean = Dense(np.prod(input_shape), activation='sigmoid')
    decoder_output_shape = Reshape(input_shape)

    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    x_decoded_mean = decoder_output_shape(x_decoded_mean)

    # VAE模型
    vae = Model(inputs, x_decoded_mean)
    encoder = Model(inputs, z_mean)

    # VAE损失
    xent_loss = binary_crossentropy(K.flatten(inputs), K.flatten(x_decoded_mean)) * np.prod(input_shape)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae, encoder

def build_mlp(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_substitute_model_using_VAE(vector, original_model, model_output_path, latent_dim=2):
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # 构建和训练VAE模型
    vae, encoder = build_vae(latent_dim, vector.shape[1:])
    vae.fit(vector, epochs=50, batch_size=32, verbose=2)  # 调整为适合你的数据集和需求的参数

    # 使用VAE编码器提取特征
    features = encoder.predict(vector)

    # 构建和训练MLP模型
    mlp = build_mlp(latent_dim)
    labels = original_model.predict(vector)  # 假设original_model能直接对原始向量进行预测
    mlp.fit(features, labels, epochs=100, batch_size=16, verbose=2)

    # 保存MLP模型
    mlp_path = os.path.join(model_output_path, 'mlp_model.h5')
    mlp.save(mlp_path)
    return "Success"

def train_substitute_Model_using_prob(vector, original_model, model_name, ori_model_name, model_output_path, type = ''):
    try:
        if not os.path.exists(model_output_path):
            os.mkdir(model_output_path)

        if model_name == 'MLP':
            # 定义MLP模型
            MLP = Sequential([
                Dense(128, activation='relu', input_shape=(vector.shape[1],)),  # input_shape 是输入特征的数量
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')  # 使用 sigmoid 激活函数进行二分类
            ])
            # 编译模型
            MLP.compile(optimizer='adam',
                        loss='mean_squared_error',  # 使用均方误差作为损失函数
                        metrics=['mean_absolute_error'])
            # 定义一个模型检查点，以保存训练过程中的最佳模型
            checkpoint = ModelCheckpoint(model_output_path + type + 'MLP_'+ori_model_name+'_prob.h5',  # 模型文件的保存路径
                                         # monitor='val_loss',  # 监控验证损失
                                         monitor='loss',  # 监控训练损失
                                         save_best_only=True,  # 仅保存最佳模型
                                         verbose=1)
            # 使用原始模型，设定标签
            label = original_model.predict_proba(vector)
            label = label[:, 1]
            MLP.fit(vector, label,
                    batch_size=16,  # 你可以根据需要调整 batch_size
                    epochs=100,  # 你可以根据需要调整 epochs 数量
                    callbacks=[checkpoint])  # 使用前面定义的检查点作为回调
            return model_output_path + type + 'MLP_'+ori_model_name+'_prob.h5'
        else:
            print('Wrong model name!')
            return None


    except Exception as e:
        print(f"An error occurred: {e}")


def train_substitute_Model(vector, original_model, model_name, ori_model_name, model_output_path, type = ''):
    try:
        if not os.path.exists(model_output_path):
            os.mkdir(model_output_path)

        if model_name == 'MLP':
            # 定义MLP模型
            MLP = Sequential([
                Dense(128, activation='relu', input_shape=(vector.shape[1],)),  # input_shape 是输入特征的数量
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')  # 使用 sigmoid 激活函数进行二分类
            ])
            # 编译模型
            MLP.compile(optimizer='adam',
                        loss='binary_crossentropy',  # 二元交叉熵作为损失函数
                        metrics=['accuracy'])
            # 定义一个模型检查点，以保存训练过程中的最佳模型
            checkpoint = ModelCheckpoint(model_output_path + type + 'MLP_'+ori_model_name+'.h5',  # 模型文件的保存路径
                                         # monitor='val_loss',  # 监控验证损失
                                         monitor='loss',  # 监控训练损失
                                         save_best_only=True,  # 仅保存最佳模型
                                         verbose=1)
            # 使用原始模型，设定标签
            label = original_model.predict(vector)
            MLP.fit(vector, label,
                    batch_size=16,  # 你可以根据需要调整 batch_size
                    epochs=100,  # 你可以根据需要调整 epochs 数量
                    callbacks=[checkpoint])  # 使用前面定义的检查点作为回调
            return model_output_path + type + 'MLP_'+ori_model_name+'.h5'
        else:
            print('Wrong model name!')
            return None


    except Exception as e:
        print(f"An error occurred: {e}")

        
def finetune_model(model, vector, label, model_output_path, type=''):
    try:
        if not os.path.exists(model_output_path):
            os.mkdir(model_output_path)

        # 编译模型
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',  # 二元交叉熵作为损失函数
                    metrics=['accuracy'])
        # 定义一个模型检查点，以保存训练过程中的最佳模型
        checkpoint = ModelCheckpoint(model_output_path + type + 'MLP_knn_1_retrained.h5',  # 模型文件的保存路径
                                     # monitor='val_loss',  # 监控验证损失
                                     monitor='loss',  # 监控训练损失
                                     save_best_only=True,  # 仅保存最佳模型
                                     verbose=1)
        # 训练模型
        model.fit(vector, label,
                batch_size=16,  # 你可以根据需要调整 batch_size
                epochs=100,  # 你可以根据需要调整 epochs 数量
                callbacks=[checkpoint])  # 使用前面定义的检查点作为回调
        return model_output_path + type + 'MLP.h5'

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def train_model(vector, label, model_name, model_output_path, type = ''):
    #vector和label都是numpy格

    try:
        if not os.path.exists(model_output_path):
            os.mkdir(model_output_path)

        print(model_output_path)
        if model_name == 'KNN_1':
            knn1 = KNeighborsClassifier(n_neighbors=1, n_jobs = 2)
            knn1.fit(vector, label)
            joblib.dump(knn1, model_output_path + type + 'knn_1.pkl')
            print('knn_1.pkl saved')
            return model_output_path + 'knn_1.pkl'
        elif model_name == 'KNN_3':
            knn3 = KNeighborsClassifier(n_neighbors=3, n_jobs = 2)
            knn3.fit(vector, label)
            joblib.dump(knn3, model_output_path + type + 'knn_3.pkl')
            return model_output_path + 'knn_3.pkl'
        elif model_name == 'KNN_5':
            knn5 = KNeighborsClassifier(n_neighbors=5, n_jobs = 2)
            knn5.fit(vector, label)
            joblib.dump(knn5, model_output_path + type + 'knn_5.pkl')
            return model_output_path + 'knn_5.pkl'
        elif model_name == 'KNN_10':
            knn5 = KNeighborsClassifier(n_neighbors=10, n_jobs = 2)
            knn5.fit(vector, label)
            joblib.dump(knn5, model_output_path + type + 'knn_10.pkl')
            return model_output_path + 'knn_10.pkl'
        elif model_name == 'MLP':
            # 定义MLP模型
            MLP = Sequential([
                Dense(128, activation='relu', input_shape=(vector.shape[1],)),  # input_shape 是输入特征的数量
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')  # 使用 sigmoid 激活函数进行二分类
            ])
            # 编译模型
            MLP.compile(optimizer='adam',
                        loss='binary_crossentropy',  # 二元交叉熵作为损失函数
                        metrics=['accuracy'])
            # 定义一个模型检查点，以保存训练过程中的最佳模型
            checkpoint = ModelCheckpoint(model_output_path + type + 'MLP.h5',  # 模型文件的保存路径
                                         # monitor='val_loss',  # 监控验证损失
                                         monitor='loss',  # 监控训练损失
                                         save_best_only=True,  # 仅保存最佳模型
                                         verbose=1)
            # 训练模型
            MLP.fit(vector, label,
                    batch_size=16,  # 你可以根据需要调整 batch_size
                    epochs=100,  # 你可以根据需要调整 epochs 数量
                    callbacks=[checkpoint])  # 使用前面定义的检查点作为回调
            return model_output_path + type + 'MLP.h5'
        elif model_name == 'SVM':
            # 定义 SVM 模型
            svmModel = svm.SVC(kernel='rbf', C=1, gamma='scale', probability=True)
            # 训练模型
            svmModel.fit(vector, label)
            # 保存模型
            joblib.dump(svmModel, model_output_path + type + 'SVM.pkl')
            return model_output_path + 'SVM.pkl'
        elif model_name == 'RandomForest':
            # 定义随机森林模型
            rf = RandomForestClassifier(max_depth=3, random_state=0)
            # rf = RandomForestClassifier()
            # 训练模型
            rf.fit(vector, label)
            # 保存模型
            joblib.dump(rf, model_output_path + type + 'RandomForest.pkl')
            return model_output_path + 'RandomForest.pkl'
        elif model_name == 'PA1':
            pa1 = PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3)
            # 训练模型
            pa1.fit(vector, label)
            # 保存模型
            joblib.dump(pa1, model_output_path + type + 'PA1.pkl')
            return model_output_path + type + 'PA1.pkl'
        elif model_name == 'AdaBoost':
            from sklearn.ensemble import AdaBoostClassifier

            # ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), random_state=123)
            # ada = ada.fit(vector, label)

            ada = AdaBoostClassifier(n_estimators=100, random_state=0)
            # # 训练模型
            ada.fit(vector, label)
            # 保存模型
            joblib.dump(ada, model_output_path + type + 'AdaBoost.pkl')
            return model_output_path + type + 'AdaBoost.pkl'
        else:
            print('Wrong model name!')
            return None


    except Exception as e:
        print(f"An error occurred: {e}")


def retrain_model(model, new_train_set, model_name, ratio, type, model_output_path, epochs=100, batch_size=16):
    """
    重新训练指定的模型。

    Args:
        model: 已经训练的模型对象。
        new_train_set: 一个包含新训练数据的元组 (vector, label)，其中
            vector 是新的训练数据特征，label 是对应的标签。
        model_output_path: 模型保存路径。
        epochs: 训练的轮数（仅对神经网络模型适用）。
        batch_size: 训练的批大小（仅对神经网络模型适用）。

    Returns:
        str: 更新后的模型保存路径。
    """
    vector, label = new_train_set

    # 确保输出路径存在
    if not os.path.exists(model_output_path):
        os.mkdir(model_output_path)

    # 检查模型类型并应用相应的训练过程
    if isinstance(model, (KNeighborsClassifier, RandomForestClassifier)):
        # 适用于 KNN、RandomForest 或其他 scikit-learn 模型
        model.fit(vector, label)
        model_file = os.path.join(model_output_path, type + model_name + str(ratio) + 'updated.pkl')
        joblib.dump(model, model_file)
    elif isinstance(model, Sequential):
        # 适用于 Keras 模型或其他支持 .fit 方法的模型
        checkpoint = ModelCheckpoint(
            os.path.join(model_output_path, type + model_name + str(ratio) + 'updated.h5'),
            monitor='val_loss',  # Assuming validation loss monitoring
            save_best_only=True,
            verbose=1
        )
        model.fit(vector, label,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[checkpoint],
                  validation_split=0.2)  # Assuming some validation
        model_file = os.path.join(model_output_path, type + model_name + str(ratio) + 'updated.h5')
    else:
        raise TypeError("Unsupported model type. The model must support the .fit method.")

    print(f'Model saved at {model_file}')
    return model_file


def dnn_train_model(X_train, Y_train, out_path):
    # 确保输出目录存在
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    seq_length = X_train.shape[1]
    X_train = X_train.reshape([-1, X_train.shape[1]])

    model = Sequential()
    model.add(Dense(300, activation='relu', use_bias=True, input_shape=(seq_length,)))
    model.add(Dense(300, activation='relu', use_bias=True))
    model.add(Dense(300, activation='relu', use_bias=True))
    model.add(Dense(300, activation='relu', use_bias=True))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    # 训练模型
    train_start_time = time.time()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir='./logs')  # 注意修改TensorBoard的日志目录
    history = model.fit(X_train, Y_train, shuffle=None, batch_size=64, epochs=40, validation_split=0.1,
                        callbacks=[tensorboard])

    train_time = time.time() - train_start_time
    print("Train time:", train_time)

    model_save_path = out_path + 'DNN.h5'
    print("save", model_save_path)
    model.save(model_save_path)
    score = model.evaluate(X_train, Y_train, batch_size=1)

    print(score)
    return model


def test_model(vector, label, model_name, model_path = './model/'):
    #vector和label都是numpy格式
    #model_path是一个文件夹
    # print(model_path)

    if os.path.isdir(model_path):
        if model_name.split('.')[-1] != 'pkl' or model_name.split('.')[-1] != 'h5':
            if model_name == 'MLP':
                model_path = model_path + model_name + '.h5'
            else:
                model_path = model_path + model_name + '.pkl'
        elif model_name.split('.')[-1] == 'pkl' or model_name.split('.')[-1] == 'h5':
            model_path = model_path + model_name
    else:#model_path是一个文件
        # print(model_path.split('.')[-1])
        if model_path.split('.')[-1] not in ['pkl', 'h5']:
            print('Wrong model path!')
            return None

    #MLP h5文件, 用tensorflow加载
    if model_name != 'MLP':
        best_model = joblib.load(model_path)
        Y_pred_probs = best_model.predict(vector)


    else:#MLP
        # 定义新模型结构，与原始模型相同，但最后一个 Dense 层没有激活函数
        MLP2 = Sequential([
            Dense(128, activation='relu', input_shape=(vector.shape[1],)),  # 同样的输入维度
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

        Y_pred_probs = MLP2.predict(vector)  # 预测概率
    #Y_pred_probs is a 2d array, then Y_pred_probs[:, 1] is the probability of being 1
    # Y_pred_probs = Y_pred_probs[:, 1]
    # if model_name == 'MLP':#MLP只输出类别为1的概率
    #     Y_pred = np.hstack([1 - Y_pred, Y_pred])


    # if label == 0:#label是0的时候，输出为1的概率，Y_pred_probs[:, ]是概率，Y_pred是类别
    #     Y_pred_probs = Y_pred_probs[:, 1]
    # elif label == 1:#label是1的时候，输出为0的概率
    #     Y_pred_probs = Y_pred_probs[:, 0]

    return Y_pred_probs

def load_test_model(model_path = '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_all/knn_1.pkl', shape = 2704):
    #vector和label都是numpy格式
    #model_path是一个文件夹
    # print(model_path)

    #MLP h5文件, 用tensorflow加载
    if 'MLP' not in model_path:
        best_model = joblib.load(model_path)
        return best_model
    else:#MLP
        # 定义新模型结构，与原始模型相同，但最后一个 Dense 层没有激活函数
        MLP2 = Sequential([
            Dense(128, activation='relu', input_shape=(shape,)),  # 同样的输入维度
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feature_extraction(file):
    important_sensitive_apis = obtain_sensitive_apis()

    vectors = []
    labels = []
    sha256 = []

    with open(file, 'r') as f:
        csv_data = csv.reader(f)

        # 读取并存储表头
        headers = next(csv_data)
        headers = headers[1:-1]

        # 确定重要特征在表头中的索引
        important_indices = [headers.index(feature) for feature in important_sensitive_apis if feature in headers]
        # 调整索引以匹配整行数据的索引
        adjusted_indices = [i + 1 for i in important_indices]

        sample_count = set()

        for line in csv_data:
            # 根据重要特征索引过滤向量
            # vector = [float(line[i]) for i in important_indices]
            if line[0] not in sample_count:
                vector = [float(line[i]) for i in adjusted_indices]
                # vector = [float(i) for i in line[1:-1]]
                label = int(float(line[-1]))
                vectors.append(vector)
                labels.append(label)
                sample_count.add(line[0])
                sha256.append(line[0])


            if len(sample_count) > 1000:
                break


        print("feature_extraction: vectors: {}, labels: {}".format(np.array(vectors).shape, np.array(labels).shape))

    return vectors, labels, sha256, headers

def feature_extraction_katz(file):
    important_sensitive_apis = obtain_sensitive_apis()

    vectors = []
    labels = []
    sha256 = []

    with open(file, 'r') as f:
        csv_data = csv.reader(f)

        # 读取并存储表头
        headers = next(csv_data)
        headers = headers[1:-1]

        # 确定重要特征在表头中的索引
        important_indices = [headers.index(feature) for feature in important_sensitive_apis if feature in headers]
        # 调整索引以匹配整行数据的索引
        adjusted_indices = [i + 1 for i in important_indices]

        sample_count = set()

        for line in csv_data:
            # 根据重要特征索引过滤向量
            # vector = [float(line[i]) for i in important_indices]
            if line[0] not in sample_count:
                vector = [float(line[i]) for i in adjusted_indices]
                # vector = [float(i) for i in line[1:-1]]
                label = int(float(line[-1]))
                vectors.append(vector)
                labels.append(label)
                sample_count.add(line[0])

            if len(sample_count) > 1000:
                break

        print("feature_extraction: vectors: {}, labels: {}".format(np.array(vectors).shape, np.array(labels).shape))

    return vectors, labels, sha256, headers

    # with open(file, 'r') as f:
    #     csv_data = csv.reader(f)
    #     # heads = next(csv_data)
    #     for line in islice(csv_data, 1, None):
    #         vector = [float(i) for i in line[1:-1]]
    #         label = int(float(line[-1]))
    #         vectors.append(vector)
    #         labels.append(label)
    #         sha256.append(line[0])
    # 
    # return vectors, labels, sha256

def get_1feature_data(file_name):
    # 断言，file_name是一个csv文件
    assert file_name.split('.')[-1] == 'csv'
    feature_csv = file_name
    # if 'katz' in file_name:
    #     vectors, labels, sha256, headers = feature_extraction_katz(feature_csv)
    # else:
    vectors, labels, sha256, headers = feature_extraction(feature_csv)
    print("vectors: {}, labels: {}".format(len(vectors), len(labels)))
    return vectors, labels, sha256, headers

def get_4features_data(train_degree_file, train_katz_file, train_closeness_file, train_harmonic_file):
    train_degree_vectors, train_degree_labels, train_degree_sha256, train_degree_headers = get_1feature_data(train_degree_file)
    train_katz_vectors, train_katz_labels, train_katz_sha256, train_degree_headers = get_1feature_data(train_katz_file)
    train_closeness_vectors, train_closeness_labels, train_closeness_sha256, train_degree_headers = get_1feature_data(train_closeness_file)
    train_harmonic_vectors, train_harmonic_labels, train_harmonic_sha256, train_degree_headers = get_1feature_data(train_harmonic_file)

    # concat 4 features
    train_vectors = []
    headers = train_degree_headers + train_degree_headers + train_degree_headers + train_degree_headers
    for i in range(len(train_degree_vectors)):
        train_vectors.append(train_degree_vectors[i] + train_katz_vectors[i] + train_closeness_vectors[i] + train_harmonic_vectors[i])

    train_labels = train_degree_labels
    train_sha256 = train_degree_sha256
    return train_vectors, train_labels, train_sha256, headers

def get_avgfeatures_data(train_degree_file, train_katz_file, train_closeness_file, train_harmonic_file):
    train_degree_vectors, train_degree_labels, train_degree_sha256, train_degree_headers = get_1feature_data(train_degree_file)
    train_katz_vectors, train_katz_labels, train_katz_sha256, train_degree_headers = get_1feature_data(train_katz_file)
    train_closeness_vectors, train_closeness_labels, train_closeness_sha256, train_degree_headers = get_1feature_data(train_closeness_file)
    train_harmonic_vectors, train_harmonic_labels, train_harmonic_sha256, train_degree_headers = get_1feature_data(train_harmonic_file)

    # avgerage 4 features
    train_vectors = []
    print("train_degree_vectors:", np.array(train_degree_vectors).shape)
    headers = train_degree_headers + train_degree_headers + train_degree_headers + train_degree_headers
    for i in range(len(train_degree_vectors)):
        avg_feature = cal_4features_avg(train_degree_vectors[i], train_katz_vectors[i], train_closeness_vectors[i], train_harmonic_vectors[i])
        train_vectors.append(avg_feature)

    print("train_vectors:", np.array(train_vectors).shape)
    train_labels = train_degree_labels
    train_sha256 = train_degree_sha256
    return train_vectors, train_labels, train_sha256, headers

def obtain_fcg_feature(sha256, year, label, feature_type='all'):
    if label == 1:
        if year == '2022':
            fcg_path = f'/data/b/shiwensong/dataset/virusshare2022_2_gexf/{sha256}.gexf'
        elif year == '2023':
            fcg_path = f'/data/b/shiwensong/dataset/virusshare2023_3_gexf/{sha256}.gexf'
        else:
            fcg_path = f'/data/b/shiwensong/dataset/virusshare{year}_gexf/{sha256}.gexf'
    else:
        fcg_path = f'/data/b/shiwensong/dataset/androzoo_benign_{year}_gexf/{sha256}.gexf'
    fcg = FCG(fcg_path, label, [0] * 430)
    G = fcg.original_call_graph
    if feature_type!= 'in_out_degree':
        fcg.cal_centralities(feature_type)

    feature_aggregators = {
        'degree': fcg.degree_feature,
        'katz': fcg.katz_feature,
        'closeness': fcg.closeness_feature,
        'harmonic': fcg.harmonic_feature
    }

    if feature_type == 'all':
        degree_feature_array = np.array(fcg.degree_feature)
        katz_feature_array = np.array(fcg.katz_feature)
        closeness_feature_array = np.array(fcg.closeness_feature)
        harmonic_feature_array = np.array(fcg.harmonic_feature)
        all_features = np.hstack([degree_feature_array, katz_feature_array, closeness_feature_array, harmonic_feature_array])
    elif feature_type == 'avg':
        all_features = cal_4features_avg(fcg.degree_feature, fcg.katz_feature, fcg.closeness_feature,
                                         fcg.harmonic_feature)
    elif feature_type == 'in_out_degree':
        # 创建节点名称到整数索引的映射
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}

        # 使用映射更新边索引
        edges = torch.tensor([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()],
                             dtype=torch.long).t().contiguous()

        # 创建 PyG 数据对象
        data = from_networkx(G)
        data.edge_index = edges

        # 计算每个节点的入度和出度
        in_degrees = torch.tensor([G.in_degree(node) for node in G.nodes()], dtype=torch.float)
        out_degrees = torch.tensor([G.out_degree(node) for node in G.nodes()], dtype=torch.float)

        # 组合入度和出度作为节点特征
        node_features = torch.stack([in_degrees, out_degrees], dim=1)
        data.x = node_features
        return  data.edge_index, node_features
    else:
        all_features = feature_aggregators[feature_type]

    node_features = torch.zeros(len(fcg.current_call_graph.nodes))
    for i in range(len(fcg.sensitive_nodes)):
        if fcg.sensitive_nodes[i] != -1:
            # sensitive node
            node_id = fcg.sensitive_nodes[i]
            # find feature value
            feature_value = None
            if feature_type == 'all':
                feature_value = all_features[i] + all_features[i + len(fcg.sensitive_nodes)] + all_features[
                    i + 2 * len(fcg.sensitive_nodes)] + all_features[i + 3 * len(fcg.sensitive_nodes)]
            else:
                feature_value = all_features[i]

            node_features[node_id] = torch.tensor(feature_value)

    return data.edge_index, node_features

# def obtain_fcg_feature(sha256, year, label, feature_type = 'all'):
#     if label == 1:
#         if year == '2022':
#             fcg_path = f'/data/b/shiwensong/dataset/virusshare2022_2_gexf/{sha256}.gexf'
#         elif year == '2023':
#             fcg_path = f'/data/b/shiwensong/dataset/virusshare2023_3_gexf/{sha256}.gexf'
#         else:
#             fcg_path = f'/data/b/shiwensong/dataset/virusshare{year}_gexf/{sha256}.gexf'
#     else:
#         fcg_path = f'/data/b/shiwensong/dataset/androzoo_benign_{year}_gexf/{sha256}.gexf'
#
#     fcg = FCG(fcg_path, label, [0] * 430)
#     # edge info
#     data = from_networkx(fcg.current_call_graph)
#
#     fcg.cal_centralities(feature_type)
#     all_features = None
#     if feature_type == 'all':
#         all_features = fcg.degree_feature + fcg.katz_feature + fcg.closeness_feature + fcg.harmonic_feature
#     elif feature_type == 'degree':
#         all_features = fcg.degree_feature
#     elif feature_type == 'katz':
#         all_features = fcg.katz_feature
#     elif feature_type == 'closeness':
#         all_features = fcg.closeness_feature
#     elif feature_type == 'harmonic':
#         all_features = fcg.harmonic_feature
#     else:
#         #average
#         all_features = cal_4features_avg(fcg.degree_feature, fcg.katz_feature, fcg.closeness_feature, fcg.harmonic_feature)
#
    # node_features = torch.zeros(len(fcg.current_call_graph.nodes))
    # for i in range(len(fcg.sensitive_nodes)):
    #     if fcg.sensitive_nodes[i] != -1:
    #         # sensitive node
    #         node_id = fcg.sensitive_nodes[i]
    #         # find feature value
    #         feature_value = None
    #         if feature_type == 'all':
    #             feature_value = all_features[i] + all_features[i + len(fcg.sensitive_nodes)] + all_features[
    #                 i + 2 * len(fcg.sensitive_nodes)] + all_features[i + 3 * len(fcg.sensitive_nodes)]
    #         else:
    #             feature_value = all_features[i]
    #
    #         node_features[node_id] = torch.tensor(feature_value)
#
#     return data.edge_index, node_features



def obtain_gcn_malware_dataset(model=None, feature_type='in_out_degree'):
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    all_edges = []
    all_node_features = []
    all_labels = []

    with ThreadPoolExecutor(max_workers=80) as executor:
        for year in years:
            dataset_file = f'{dir_path}degree_1_{year}.csv'
            vectors, labels, sha256, headers = get_1feature_data(dataset_file)
            sha256_train = sha256[:int(len(sha256) * 0.2)]
            vectors = vectors[:int(len(vectors) * 0.2)]
            futures = [executor.submit(obtain_fcg_feature, sha, year, 1, feature_type) for sha in sha256_train]
            results = [future.result() for future in futures]

            if model:
                y_pre_labels = model.predict(vectors)
            else:
                y_pre_labels = labels[:int(len(labels) * 0.2)]

            all_labels.extend(y_pre_labels)

            for edge, node_feature in results:
                all_edges.append(edge)
                all_node_features.append(node_feature)

    print("all_edges:", len(all_edges))
    print("all_node_features:", len(all_node_features))
    print("all_label", len(all_labels))
    # Save the data as a pickle file
    with open('malware_dataset.pkl', 'wb') as f:
        pickle.dump((all_edges, all_node_features, all_labels), f)

    return all_edges, all_node_features, all_labels
    # def get_1feature_data(file_name):
    #     # 断言，file_name是一个csv文件
    #     assert file_name.split('.')[-1] == 'csv'
    #     feature_csv = file_name
    #     # if 'katz' in file_name:
    #     #     vectors, labels, sha256, headers = feature_extraction_katz(feature_csv)
    #     # else:
    #     vectors, labels, sha256, headers = feature_extraction(feature_csv)
    #     print("vectors: {}, labels: {}".format(len(vectors), len(labels)))
    #     return vectors, labels, sha256, headers

def obtain_gcn_benign_dataset(model=None, feature_type = 'in_out_degree'):
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    all_edges = []
    all_node_features = []
    all_labels = []

    with ThreadPoolExecutor(max_workers=80) as executor:
        for year in years:
            dataset_file = f'{dir_path}degree_0_{year}.csv'
            vectors, labels, sha256, headers = get_1feature_data(dataset_file)
            sha256_train = sha256[:int(len(sha256) * 0.2)]
            vectors = vectors[:int(len(vectors) * 0.2)]
            futures = [executor.submit(obtain_fcg_feature, sha, year, 0, feature_type) for sha in sha256_train]
            results = [future.result() for future in futures]
            
            if model:
                y_pre_labels = model.predict(vectors)
            else:
                y_pre_labels = labels[:int(len(labels) * 0.2)]

            all_labels.extend(y_pre_labels)

            for edge, node_feature in results:
                all_edges.append(edge)
                all_node_features.append(node_feature)


    print("all_edges:", len(all_edges))
    print("all_node_features:", len(all_node_features))
    print("all_label", len(all_labels))
    # Save the data as a pickle file
    with open('baseline_benign_dataset.pkl', 'wb') as f:
        pickle.dump((all_edges, all_node_features, all_labels), f)

    return all_edges, all_node_features, all_labels

def cal_4features_avg(degree_vector, katz_vector, closeness_vector, harmonic_vector):
    avg_vector = []
    for i in range(len(degree_vector)):
        avg_feature = (degree_vector[i] + katz_vector[i] + closeness_vector[i] + harmonic_vector[i]) / 4.0
        avg_vector.append(avg_feature)
    return avg_vector


def get_2features_data(train_degree_file, train_katz_file):
    train_degree_vectors, train_degree_labels, train_degree_sha256 = get_1feature_data(train_degree_file)
    train_katz_vectors, train_katz_labels, train_katz_sha256 = get_1feature_data(train_katz_file)

    # concat 2 features
    train_vectors = []
    for i in range(len(train_degree_vectors)):
        train_vectors.append(train_degree_vectors[i] + train_katz_vectors[i])
    train_labels = train_degree_labels
    train_sha256 = train_degree_sha256
    return train_vectors, train_labels, train_sha256

def find_important_features_based_on_frequency(vector, select_num=300):
    # 步骤 1 和 2: 找出所有样本中非零的特征并计算出现频次
    feature_counts = np.sum(vector != 0, axis=0)
    # print("feature_counts:", feature_counts.shape)

    # # 步骤 3: 排序并选择前 300 个特征
    # # 步骤 3: 排序并选择前 300 个特征
    # top_300_indices = np.argsort(feature_counts)[::-1][:300]
    #
    # # 步骤 4: 打印出这些特征在原始向量中的下标
    # # print("Top 300 feature indices:", top_300_indices)
    #
    # # 步骤 4: 获取这 300 个特征对应的值
    # # important_features_values = vector[:, top_300_indices]
    #
    # # print("important_features_values:", important_features_values)
    #
    # return top_300_indices

    # 对特征出现频次进行排序和阈值计算
    feature_counts_sorted = sorted(feature_counts, reverse=True)
    border_top = feature_counts_sorted[int(len(feature_counts_sorted) * 0.1)]
    border_bottom = feature_counts_sorted[int(len(feature_counts_sorted) * 0.9)]

    feature_indexes = []

    dimension = vector.shape[1]

    for index in range(dimension):
        feature_freq = feature_counts[index]
        if border_bottom <= feature_freq <= border_top:
            feature_indexes.append(index)

    # 选择特征
    top_features = sorted(feature_indexes, key=lambda idx: feature_counts[idx], reverse=True)[:select_num]

    return top_features


def find_important_features_based_on_sum(vectors, select_num=300):
    # 计算良性和恶意样本中每个特征的总和
    feature_sums = vectors.sum(axis=0)

    # 对特征总和进行排序和阈值计算
    feature_top = sorted(feature_sums, reverse=True)
    feature_top = [item for item in feature_top if item > 0]
    border_feature_top = feature_top[int(len(feature_top) * 0.1)]
    border_feature_bottom = feature_top[int(len(feature_top) * 0.9)]
    # print("border_feature_top:", border_feature_top)
    # print("border_feature_bottom:", border_feature_bottom)

    feature_indexes = []

    dimension = vectors.shape[1]

    for index in range(dimension):
        feature_int = feature_sums[index]
        # print("feature_int:", feature_int)
        if feature_int >= border_feature_bottom and feature_int <= border_feature_top:
            feature_indexes.append(index)


    # 选择特征
    feature_top_fts = sorted(feature_indexes, key=lambda idx: feature_sums[idx], reverse=True)[:select_num]

    # print("feature_top_fts:", feature_top_fts)
    # print("feature_top_fts:", len(feature_top_fts))
    return feature_top_fts

def obtain_one_year_dataset_one_feature(dir_path, year, label, type):
    # dir_path = '/data/a/shiwensong/dataset/features/'
    # year = 2018
    # label = 0
    dataset_file = dir_path + type+ '_{}_{}.csv'.format(label, year)
    X_train, Y_train, sha256 = get_1feature_data(dataset_file)
    return X_train, Y_train

def obtain_one_year_dataset_4features(dir_path, year, label):
    # dir_path = '/data/a/shiwensong/dataset/features/'
    # year = 2018
    # label = 0
    dataset_degree_file = dir_path + 'degree_{}_{}.csv'.format(label, year)
    dataset_katz_file = dir_path + 'katz_{}_{}.csv'.format(label, year)
    dataset_harmonic_file = dir_path + 'harmonic_{}_{}.csv'.format(label, year)
    dataset_closeness_file = dir_path + 'closeness_{}_{}.csv'.format(label, year)

    X_train, Y_train, sha256, headers = get_4features_data(dataset_degree_file, dataset_katz_file, dataset_closeness_file, dataset_harmonic_file)
    length = int(len(X_train) * 0.8)
    X_train = X_train[:length]
    
    #random choose 100 samples
    random_numbers = random.sample(range(len(X_train)), 100)
    X_train = [X_train[i] for i in random_numbers]
    Y_train = Y_train[:100]
    
    return X_train, Y_train

def obtain_manyyears_features(label, type):
    dir_path = '/data/a/shiwensong/dataset/feature_Nov30/'

    # all dataset

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for year in ['2018', '2022', '2023']:
        X_dataset, Y_dataset = obtain_one_year_dataset_one_feature(dir_path, year, label, type)
        #分成训练集和测试集
        length_80_percent = int(0.8 * len(X_dataset))
        X_train_one_year = X_dataset[:length_80_percent]
        Y_train_one_year = Y_dataset[:length_80_percent]
        X_test_one_year = X_dataset[length_80_percent:]
        Y_test_one_year = Y_dataset[length_80_percent:]
        print("X_train: {}, Y_train: {}".format(np.array(X_train_one_year).shape, np.array(Y_train_one_year).shape))
        print("X_test: {}, Y_test: {}".format(np.array(X_test_one_year).shape, np.array(Y_test_one_year).shape))
        X_train.extend(X_train_one_year)
        Y_train.extend(Y_train_one_year)
        X_test.extend(X_test_one_year)
        Y_test.extend(Y_test_one_year)
    
    return X_train, Y_train, X_test, Y_test

def find_important_features_based_on_shapeley(X_train, Y_train, select_num=1200):
    # 选择特征
    clf = RandomForestClassifier(max_depth=10, min_samples_split=4, min_samples_leaf=2)
    clf.fit(X_train, Y_train)
    #explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    # print("shap_values:", shap_values)
    # shap_matrix = np.array(shap_values)
    # shap.plots.waterfall(shap_values[0])
    # shap.plots.beeswarm(shap_values)


def find_important_features_based_on_decisiontree(X_train, Y_train, select_num=1000):
    # 选择特征
    clf = DecisionTreeClassifier(max_depth=10, min_samples_split=4, min_samples_leaf=2)
    clf.fit(X_train, Y_train)
    feature_importances = clf.feature_importances_
    print("feature_importances:", feature_importances)

    feature_top_fts = sorted(range(len(feature_importances)), key=lambda idx: feature_importances[idx], reverse=True)[:select_num]
    print("feature_top_fts:", feature_top_fts)
    print(feature_importances[feature_top_fts[:select_num]])
    # print(feature_importances[:20])

    # # 可视化top300特征（假设 obtain_sensitive_apis 返回相关特征名列表）
    # degree_feature_name = obtain_sensitive_apis()
    # katz_feature_name = obtain_sensitive_apis()
    # closeness_feature_name = obtain_sensitive_apis()
    # harmonic_feature_name = obtain_sensitive_apis()
    # feature_names = degree_feature_name + katz_feature_name + closeness_feature_name + harmonic_feature_name
    # feature_top_fts_name = [feature_names[i] for i in feature_top_fts]
    # print("feature_top_fts_name:", feature_top_fts_name)
    # # 用plt画图
    # plt.figure(figsize=(20, 20))
    # plt.barh(feature_top_fts_name, feature_importances[feature_top_fts])
    # plt.xlabel('Feature Importance')
    # plt.ylabel('Feature Name')
    #
    # # explain the model's predictions using SHAP
    # explainer = shap.TreeExplainer(clf)
    # shap_values = explainer.shap_values(X_train)
    # print("shap_values:", shap_values)
    # print("shap_values:", np.array(shap_values).shape)
    # print(shap_values[0])
    # shap_matrix = np.array(shap_values)
    # shap.plots.waterfall(shap_values[0])
    # shap.plots.beeswarm(shap_values)

    return feature_top_fts



def benign_dataset():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_degree_file = f'{dir_path}degree_0_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_0_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_0_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(0.8 * len(Y_train_benign))
        X_train_benign = X_train_benign[:length_80_percent_benign_2018]
        Y_train_benign = Y_train_benign[:length_80_percent_benign_2018]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def test_benign_dataset():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_degree_file = f'{dir_path}degree_0_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_0_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_0_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(0.8 * len(Y_train_benign))
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        Y_train_benign = Y_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def val_benign_dataset():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_degree_file = f'{dir_path}degree_0_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_0_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_0_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_90_percent_benign_2018:]
        Y_train_benign = Y_train_benign[length_90_percent_benign_2018:]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def benign_dataset_avg_feature():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_degree_file = f'{dir_path}degree_0_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_0_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_0_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_avgfeatures_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(0.8 * len(Y_train_benign))
        X_train_benign = X_train_benign[:length_80_percent_benign_2018]
        Y_train_benign = Y_train_benign[:length_80_percent_benign_2018]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def test_benign_dataset_avg_feature():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_degree_file = f'{dir_path}degree_0_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_0_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_0_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_avgfeatures_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(0.8 * len(Y_train_benign))
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        Y_train_benign = Y_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def val_benign_dataset_avg_feature():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_degree_file = f'{dir_path}degree_0_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_0_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_0_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_avgfeatures_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_90_percent_benign_2018:]
        Y_train_benign = Y_train_benign[length_90_percent_benign_2018:]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train


def test_benign_dataset_1feature(type = 'degree'):
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_file = f'{dir_path}{type}_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_1feature_data(dataset_benign_file)
        length_80_percent_benign_2018 = int(0.8 * len(Y_train_benign))
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        Y_train_benign = Y_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def val_benign_dataset_1feature(type = 'degree'):
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_file = f'{dir_path}{type}_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_1feature_data(dataset_benign_file)
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_90_percent_benign_2018:]
        Y_train_benign = Y_train_benign[length_90_percent_benign_2018:]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def benign_dataset_1feature(type = 'degree'):
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_file = f'{dir_path}{type}_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_1feature_data(dataset_benign_file)
        length_80_percent_benign_2018 = int(0.8 * len(Y_train_benign))
        X_train_benign = X_train_benign[:length_80_percent_benign_2018]
        Y_train_benign = Y_train_benign[:length_80_percent_benign_2018]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def malware_dataset():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        num = nums[i]
        dataset_benign_degree_file = f'{dir_path}degree_1_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_1_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_1_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(len(Y_train_benign)*0.8)
        X_train_benign = X_train_benign[:length_80_percent_benign_2018]
        Y_train_benign = Y_train_benign[:length_80_percent_benign_2018]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def test_malware_dataset():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        num = nums[i]
        dataset_benign_degree_file = f'{dir_path}degree_1_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_1_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_1_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(len(Y_train_benign)*0.8)
        length_90_percent_benign_2018 = int(len(Y_train_benign)*0.9)
        X_train_benign = X_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        Y_train_benign = Y_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def val_malware_dataset():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        num = nums[i]
        dataset_benign_degree_file = f'{dir_path}degree_1_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_1_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_1_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_90_percent_benign_2018 = int(len(Y_train_benign)*0.9)
        X_train_benign = X_train_benign[length_90_percent_benign_2018:]
        Y_train_benign = Y_train_benign[length_90_percent_benign_2018:]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def malware_dataset_avg_feature():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        num = nums[i]
        dataset_benign_degree_file = f'{dir_path}degree_1_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_1_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_1_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_avgfeatures_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(len(X_train_benign)*0.8)
        X_train_benign = X_train_benign[:length_80_percent_benign_2018]
        Y_train_benign = Y_train_benign[:length_80_percent_benign_2018]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def test_malware_dataset_avg_feature():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        num = nums[i]
        dataset_benign_degree_file = f'{dir_path}degree_1_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_1_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_1_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_avgfeatures_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(len(X_train_benign)*0.8)
        length_90_percent_benign_2018 = int(len(X_train_benign)*0.9)
        X_train_benign = X_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        Y_train_benign = Y_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def val_malware_dataset_avg_feature():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        num = nums[i]
        dataset_benign_degree_file = f'{dir_path}degree_1_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_1_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_1_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_avgfeatures_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_90_percent_benign_2018 = int(len(X_train_benign)*0.9)
        X_train_benign = X_train_benign[length_90_percent_benign_2018:]
        Y_train_benign = Y_train_benign[length_90_percent_benign_2018:]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def malware_dataset_1feature(type = 'degree'):
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        num = nums[i]
        dataset_benign_file = f'{dir_path}{type}_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_1feature_data(dataset_benign_file)
        length_80_percent_benign_2018 = int(len(Y_train_benign)*0.8)
        X_train_benign = X_train_benign[:length_80_percent_benign_2018]
        Y_train_benign = Y_train_benign[:length_80_percent_benign_2018]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def test_malware_dataset_1feature(type = 'degree'):
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        num = nums[i]
        dataset_benign_file = f'{dir_path}{type}_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_1feature_data(dataset_benign_file)
        length_80_percent_benign_2018 = int(len(Y_train_benign)*0.8)
        length_90_percent_benign_2018 = int(len(Y_train_benign)*0.9)
        X_train_benign = X_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        Y_train_benign = Y_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def val_malware_dataset_1feature(type = 'degree'):
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        num = nums[i]
        dataset_benign_file = f'{dir_path}{type}_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_1feature_data(dataset_benign_file)
        length_90_percent_benign_2018 = int(len(Y_train_benign)*0.9)
        X_train_benign = X_train_benign[length_90_percent_benign_2018:]
        Y_train_benign = Y_train_benign[length_90_percent_benign_2018:]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def get_index_from_knn1_benign_malware(vector, benign_model_path = './430features_3yearsdataset_benign/knn_1.pkl', malware_model_path = './430features_3yearsdataset_malware/knn_1.pkl'):
    benign_model = joblib.load(benign_model_path)
    # 获取最近邻居的索引
    benign_distances, benign_indices = benign_model.kneighbors(vector)
    malware_model = joblib.load(malware_model_path)
    # 获取最近邻居的索引
    malware_distances, malware_indices = malware_model.kneighbors(vector)
    return benign_indices[0][0], malware_indices[0][0]

def get_distance_from_knn1(vector, benign_model_path = './430features_3yearsdataset_benign/knn_1.pkl', malware_model_path = './430features_3yearsdataset_malware/knn_1.pkl'):
    print("get_distance_from_knn1")
    benign_model = joblib.load(benign_model_path)
    # 获取最近邻居的索引
    benign_distances, benign_indices = benign_model.kneighbors(vector)
    print(benign_distances)
    print(benign_indices)
    
    malware_model = joblib.load(malware_model_path)
    # 获取最近邻居的索引
    malware_distances, malware_indices = malware_model.kneighbors(vector)
    print(malware_distances)
    print(malware_indices)

    return benign_distances[0][0], malware_distances[0][0]
    # return benign_distances[0][0]


def get_distance_from_knn5(vector, benign_model_path='./430features_3yearsdataset_benign/knn_5.pkl',
                           malware_model_path='./430features_3yearsdataset_malware/knn_5.pkl'):
    print("get_distance_from_knn5")
    benign_model = joblib.load(benign_model_path)
    # 获取最近邻居的索引
    benign_distances, benign_indices = benign_model.kneighbors(vector)
    print(benign_distances)
    print(benign_indices)
    #计算每个样本的平均距离
    benign_mean_distance = np.mean(benign_distances, axis=1)
    print(benign_mean_distance)

    malware_model = joblib.load(malware_model_path)
    # 获取最近邻居的索引
    malware_distances, malware_indices = malware_model.kneighbors(vector)
    print(malware_distances)
    print(malware_indices)
    # 计算每个样本的平均距离
    malware_mean_distance = np.mean(malware_distances, axis=1)
    print(malware_mean_distance)

    return benign_mean_distance[0], malware_mean_distance[0]

def get_distance_from_knn10(vector, model_path = './430features_3yearsdataset_all/knn_10.pkl'):
    print("get_distance_from_knn10")
    model = joblib.load(model_path)
    # 获取最近邻居的索引
    distances, indices = model.kneighbors(vector)
    print(distances)
    print(indices)

    return distances[0][0]

def get_distance_from_knn1_all(vector, model_path = './430features_3yearsdataset_all/knn_1.pkl'):
    print("get_distance_from_knn10")
    model = joblib.load(model_path)
    # 获取最近邻居的索引
    distances, indices = model.kneighbors(vector)
    print(distances)
    print(indices)

    return distances[0][0]

def get_distance_from_original_knn1(vector, model_path = '/data/c/shiwensong/ssw/MalScan-code/model/knn_model_1_4features.pkl'):
    model = joblib.load(model_path)
    # 获取最近邻居的索引
    distances, indices = model.kneighbors(vector)
    print(distances)
    print(indices)
    Y_pred = model.predict(vector)

    return distances[0][0], Y_pred[0]

def test_dataset_benign():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_degree_file = f'{dir_path}degree_0_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_0_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_0_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(0.8 * len(Y_train_benign))
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        Y_train_benign = Y_train_benign[length_80_percent_benign_2018:length_90_percent_benign_2018]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def val_dataset_benign():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    X_train = []
    Y_train = []

    for year in years:
        dataset_benign_degree_file = f'{dir_path}degree_0_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_0_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_0_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_0_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_90_percent_benign_2018:]
        Y_train_benign = Y_train_benign[length_90_percent_benign_2018:]
        print('benign_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('benign_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("benign dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def test_dataset_malware():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    # nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        dataset_benign_degree_file = f'{dir_path}degree_1_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_1_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_1_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_80_percent_benign_2018 = int(0.8 * len(Y_train_benign))
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_80_percent_benign_2018: length_90_percent_benign_2018]
        Y_train_benign = Y_train_benign[length_80_percent_benign_2018: length_90_percent_benign_2018]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def val_dataset_malware():
    dir_path = '/data/b/shiwensong/dataset/feature_Nov30/'

    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    # nums = [3800, 2900, 2700, 3300, 2800, 3500]

    X_train = []
    Y_train = []

    for i in range(len(years)):
        year = years[i]
        dataset_benign_degree_file = f'{dir_path}degree_1_{year}.csv'
        dataset_benign_katz_file = f'{dir_path}katz_1_{year}.csv'
        dataset_benign_harmonic_file = f'{dir_path}harmonic_1_{year}.csv'
        dataset_benign_closeness_file = f'{dir_path}closeness_1_{year}.csv'
        #
        # # 每一个文件里面的数据，取前80%作为训练集，20%作为验证集
        #
        X_train_benign, Y_train_benign, benign_sha256, benign_headers = get_4features_data(
            dataset_benign_degree_file,
            dataset_benign_katz_file,
            dataset_benign_closeness_file,
            dataset_benign_harmonic_file)
        length_90_percent_benign_2018 = int(0.9 * len(Y_train_benign))
        X_train_benign = X_train_benign[length_90_percent_benign_2018:]
        Y_train_benign = Y_train_benign[length_90_percent_benign_2018:]
        print('malware_2018', np.array(X_train_benign).shape)
        X_train.extend(X_train_benign)
        Y_train.extend(Y_train_benign)
        print('malware_all', np.array(X_train).shape)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # 确保你已经有了处理好的数据 X_train, y_train
    # X_train 是特征，y_train 是标签
    print("malware dataset shape")
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

def dataset_one_feature(file_name, year, label):
    dataset_file = '/data/a/shiwensong/dataset/features/'+file_name+'_'+label+'_'+year+'.csv'
    #
    X_train, Y_train, sha256 = get_1feature_data(dataset_file)
    # length_80_percent_benign_2018 = int(0.8 * len(X_train_benign_2018))
    # X_train_benign_2018 = X_train_benign_2018[:length_80_percent_benign_2018]
    # Y_train_benign_2018 = Y_train_benign_2018[:length_80_percent_benign_2018]
    print('dataset', np.array(X_train).shape)
    return X_train, Y_train

def deal_with_benign_dataset():
    # #degree top 300 important features benign
    #2018_benign
    X_benign_2018_degree, Y_benign_2018_degree = dataset_one_feature('degree', '2018', '0')
    # #2019_benign
    X_benign_2019_degree, Y_benign_2019_degree = dataset_one_feature('degree', '2017', '0')
    # # 2020_benign
    X_benign_2020_degree, Y_benign_2020_degree = dataset_one_feature('degree', '2016', '0')
    # #2022_benign
    X_benign_2022_degree, Y_benign_2022_degree = dataset_one_feature('degree', '2015', '0')
    #2023_benign
    X_benign_2023_degree, Y_benign_2023_degree = dataset_one_feature('degree', '2014', '0')
    # 2023_benign
    X_benign_2013_degree, Y_benign_2013_degree = dataset_one_feature('degree', '2013', '0')
    #
    # #合并
    X_benign_degree = X_benign_2018_degree + X_benign_2019_degree + X_benign_2020_degree + X_benign_2022_degree + X_benign_2023_degree + X_benign_2013_degree
    # X_benign_degree = X_benign_2018_degree + X_benign_2022_degree + X_benign_2023_degree
    X_benign_degree = np.array(X_benign_degree)
    print(X_benign_degree.shape)
    #
    benign_degree_feature_indexes_frequency = find_important_features_based_on_frequency(X_benign_degree)
    print("benign degree")
    benign_degree_feature_indexes_sum = find_important_features_based_on_sum(X_benign_degree)
    # print(benign_degree_feature_indexes)
    #
    # #degree top 300 important features malware
    # 2018_malware
    X_malware_2018_degree, Y_malware_2018_degree = dataset_one_feature('degree', '2018', '1')
    # 2022_malware
    X_malware_2022_degree, Y_malware_2022_degree = dataset_one_feature('degree', '2017', '1')
    # 2023_malware
    X_bmalware_2023_degree, Y_malware_2023_degree = dataset_one_feature('degree', '2016', '1')
    # 2018_malware
    X_malware_2015_degree, Y_malware_2015_degree = dataset_one_feature('degree', '2015', '1')
    # 2022_malware
    X_malware_2014_degree, Y_malware_2014_degree = dataset_one_feature('degree', '2014', '1')
    # 2023_malware
    X_malware_2013_degree, Y_malware_2013_degree = dataset_one_feature('degree', '2013', '1')
    #
    # # 合并
    X_malware_degree = X_malware_2018_degree + X_malware_2022_degree + X_bmalware_2023_degree + X_malware_2015_degree + X_malware_2014_degree + X_malware_2013_degree
    X_malware_degree = np.array(X_malware_degree)
    # print(X_malware_degree.shape)
    #
    malware_degree_feature_indexes_frequency = find_important_features_based_on_frequency(X_malware_degree)
    print("malware degree")
    malware_degree_feature_indexes_sum = find_important_features_based_on_sum(X_malware_degree)
    # print("degree")
    # print(malware_degree_feature_indexes)



    # #katz top 300 important features
    X_benign_2018_katz, Y_benign_2018_katz = dataset_one_feature('katz', '2018', '0')
    #
    X_benign_2019_katz, Y_benign_2019_katz = dataset_one_feature('katz', '2017', '0')

    X_benign_2020_katz, Y_benign_2020_katz = dataset_one_feature('katz', '2016', '0')

    X_benign_2022_katz, Y_benign_2022_katz = dataset_one_feature('katz', '2015', '0')

    X_benign_2023_katz, Y_benign_2023_katz = dataset_one_feature('katz', '2014', '0')

    X_benign_2013_katz, Y_benign_2013_katz = dataset_one_feature('katz', '2013', '0')

    #合并
    X_benign_katz = X_benign_2018_katz + X_benign_2019_katz + X_benign_2020_katz + X_benign_2022_katz + X_benign_2023_katz + X_benign_2013_katz
    # X_benign_katz = X_benign_2018_katz + X_benign_2022_katz + X_benign_2023_katz
    X_benign_katz = np.array(X_benign_katz)
    print(X_benign_katz.shape)

    benign_katz_feature_indexes_frequency = find_important_features_based_on_frequency(X_benign_katz)
    benign_katz_feature_indexes_sum = find_important_features_based_on_sum(X_benign_katz)
    print("benign katz")

    # #katz top 300 important features malware
    # 2018_malware
    X_malware_2018_katz, Y_malware_2018_katz = dataset_one_feature('katz', '2018', '1')
    # 2022_malware
    X_malware_2022_katz, Y_malware_2022_katz = dataset_one_feature('katz', '2017', '1')
    # 2023_malware
    X_malware_2023_katz, Y_malware_2023_katz = dataset_one_feature('katz', '2016', '1')
    # 2018_malware
    X_malware_2015_katz, Y_malware_2015_katz = dataset_one_feature('katz', '2015', '1')
    # 2022_malware
    X_malware_2014_katz, Y_malware_2014_katz = dataset_one_feature('katz', '2014', '1')
    # 2023_malware
    X_malware_2013_katz, Y_malware_2013_katz = dataset_one_feature('katz', '2013', '1')

    #
    # # 合并
    X_malware_katz = X_malware_2018_katz + X_malware_2022_katz + X_malware_2023_katz + X_malware_2015_katz + X_malware_2014_katz + X_malware_2013_katz
    X_malware_katz = np.array(X_malware_katz)
    # print(X_malware_degree.shape)
    #
    malware_katz_feature_indexes_frequency = find_important_features_based_on_frequency(X_malware_katz)
    print("malware katz")
    malware_katz_feature_indexes_sum = find_important_features_based_on_sum(X_malware_katz)

    #closeness top 300 important features
    X_benign_2018_closeness, Y_benign_2018_closeness = dataset_one_feature('closeness', '2018', '0')

    X_benign_2019_closeness, Y_benign_2019_closeness = dataset_one_feature('closeness', '2017', '0')

    X_benign_2020_closeness, Y_benign_2020_closeness = dataset_one_feature('closeness', '2016', '0')

    X_benign_2022_closeness, Y_benign_2022_closeness = dataset_one_feature('closeness', '2015', '0')

    X_benign_2023_closeness, Y_benign_2023_closeness = dataset_one_feature('closeness', '2014', '0')

    X_benign_2013_closeness, Y_benign_2013_closeness = dataset_one_feature('closeness', '2013', '0')


    #合并
    X_benign_closeness = X_benign_2018_closeness +  X_benign_2022_closeness + X_benign_2023_closeness + X_benign_2019_closeness + X_benign_2020_closeness + X_benign_2013_closeness
    X_benign_closeness = np.array(X_benign_closeness)
    print(X_benign_closeness.shape)

    benign_closeness_feature_indexes_frequency = find_important_features_based_on_frequency(X_benign_closeness)
    benign_closeness_feature_indexes_sum = find_important_features_based_on_sum(X_benign_closeness)
    print("closeness")

    # closeness top 300 important features
    X_malware_2018_closeness, Y_malware_2018_closeness = dataset_one_feature('closeness', '2018', '1')

    X_malware_2022_closeness, Y_malware_2022_closeness = dataset_one_feature('closeness', '2017', '1')

    X_malware_2023_closeness, Y_malware_2023_closeness = dataset_one_feature('closeness', '2016', '1')

    X_malware_2015_closeness, Y_malware_2015_closeness = dataset_one_feature('closeness', '2015', '1')

    X_malware_2014_closeness, Y_malware_2014_closeness = dataset_one_feature('closeness', '2014', '1')

    X_malware_2013_closeness, Y_malware_2013_closeness = dataset_one_feature('closeness', '2013', '1')

    # 合并
    X_malware_closeness = X_malware_2018_closeness + X_malware_2022_closeness + X_malware_2023_closeness + X_malware_2015_closeness + X_malware_2014_closeness + X_malware_2013_closeness
    X_malware_closeness = np.array(X_malware_closeness)
    print(X_malware_closeness.shape)

    malware_closeness_feature_indexes_frequency = find_important_features_based_on_frequency(X_malware_closeness)
    malware_closeness_feature_indexes_sum = find_important_features_based_on_sum(X_malware_closeness)
    print("closeness")

    #harmonic top 300 important features
    X_benign_2018_harmonic, Y_benign_2018_harmonic = dataset_one_feature('harmonic', '2018', '0')

    X_benign_2019_harmonic, Y_benign_2019_harmonic = dataset_one_feature('harmonic', '2017', '0')

    X_benign_2020_harmonic, Y_benign_2020_harmonic = dataset_one_feature('harmonic', '2016', '0')

    X_benign_2022_harmonic, Y_benign_2022_harmonic = dataset_one_feature('harmonic', '2015', '0')

    X_benign_2023_harmonic, Y_benign_2023_harmonic = dataset_one_feature('harmonic', '2014', '0')

    X_benign_2013_harmonic, Y_benign_2013_harmonic = dataset_one_feature('harmonic', '2013', '0')

    #合并
    X_benign_harmonic = X_benign_2018_harmonic + X_benign_2019_harmonic + X_benign_2020_harmonic + X_benign_2022_harmonic + X_benign_2023_harmonic + X_benign_2013_harmonic
    # X_benign_harmonic = X_benign_2018_harmonic + X_benign_2022_harmonic + X_benign_2023_harmonic
    X_benign_harmonic = np.array(X_benign_harmonic)
    print(X_benign_harmonic.shape)

    benign_harmonic_feature_indexes_frequency = find_important_features_based_on_frequency(X_benign_harmonic)
    benign_harmonic_feature_indexes_sum = find_important_features_based_on_sum(X_benign_harmonic)
    print("harmonic")

    # harmonic top 300 important features
    X_malware_2018_harmonic, Y_malware_2018_harmonic = dataset_one_feature('harmonic', '2018', '1')

    X_malware_2022_harmonic, Y_malware_2022_harmonic = dataset_one_feature('harmonic', '2017', '1')

    X_malware_2023_harmonic, Y_malware_2023_harmonic = dataset_one_feature('harmonic', '2016', '1')

    X_malware_2015_harmonic, Y_malware_2015_harmonic = dataset_one_feature('harmonic', '2015', '1')

    X_malware_2014_harmonic, Y_malware_2014_harmonic = dataset_one_feature('harmonic', '2014', '1')

    X_malware_2013_harmonic, Y_malware_2013_harmonic = dataset_one_feature('harmonic', '2013', '1')

    # 合并
    X_malware_harmonic = X_malware_2018_harmonic + X_malware_2022_harmonic + X_malware_2023_harmonic + X_malware_2015_harmonic + X_malware_2014_harmonic + X_malware_2013_harmonic
    X_malware_harmonic = np.array(X_malware_harmonic)
    print(X_malware_harmonic.shape)

    malware_harmonic_feature_indexes_frequency = find_important_features_based_on_frequency(X_malware_harmonic)
    malware_harmonic_feature_indexes_sum = find_important_features_based_on_sum(X_malware_harmonic)
    print("harmonic")

    #
    # #四个特征的特征集合的交集
    # # 找出交集
    # intersect12 = np.intersect1d(benign_degree_feature_indexes_frequency, benign_katz_feature_indexes_frequency)
    # intersect123 = np.intersect1d(benign_degree_feature_indexes_frequency, benign_harmonic_feature_indexes_frequency)
    # intersect1234 = np.intersect1d(benign_degree_feature_indexes_frequency, benign_closeness_feature_indexes_frequency)
    # intersect12345 = np.intersect1d(benign_harmonic_feature_indexes_frequency, benign_closeness_feature_indexes_frequency)
    # print(intersect12.shape)
    # print(intersect123.shape)
    # print(intersect1234.shape)
    # print(intersect12345.shape)
    #
    # malware_intersect12 = np.intersect1d(malware_degree_feature_indexes_frequency, malware_katz_feature_indexes_frequency)
    # malware_intersect123 = np.intersect1d(malware_degree_feature_indexes_frequency, malware_harmonic_feature_indexes_frequency)
    # malware_intersect1234 = np.intersect1d(malware_degree_feature_indexes_frequency, malware_closeness_feature_indexes_frequency)
    # malware_intersect12345 = np.intersect1d(malware_harmonic_feature_indexes_frequency, malware_closeness_feature_indexes_frequency)
    # print(malware_intersect12.shape)
    # print(malware_intersect123.shape)
    # print(malware_intersect1234.shape)
    # print(malware_intersect12345.shape)
    #
    # benign_malware_ineersect12 = np.intersect1d(benign_degree_feature_indexes_frequency, malware_degree_feature_indexes_frequency)
    # benign_malware_ineersect123 = np.intersect1d(benign_harmonic_feature_indexes_frequency, malware_harmonic_feature_indexes_frequency)
    # benign_malware_ineersect1234 = np.intersect1d(benign_closeness_feature_indexes_frequency, malware_closeness_feature_indexes_frequency)
    # benign_malware_intersect1234 = np.intersect1d(benign_katz_feature_indexes_frequency, malware_katz_feature_indexes_frequency)
    # print(benign_malware_ineersect12.shape)
    # print(benign_malware_ineersect123.shape)
    # print(benign_malware_ineersect1234.shape)
    # print(benign_malware_intersect1234.shape)
    #
    # intersect12_sum = np.intersect1d(benign_degree_feature_indexes_sum, benign_katz_feature_indexes_sum)
    # intersect123_sum = np.intersect1d(intersect12_sum, benign_harmonic_feature_indexes_sum)
    # intersect1234_sum = np.intersect1d(intersect123_sum, benign_closeness_feature_indexes_sum)
    # print(intersect12_sum.shape)
    # print(intersect123_sum.shape)
    # print(intersect1234_sum.shape)
    # #
    # intersectfrequency_sum = np.intersect1d(benign_degree_feature_indexes_sum, malware_degree_feature_indexes_sum)
    # print(intersectfrequency_sum.shape)
    #
    # # 计算并集 benign
    union12_benign = np.union1d(benign_degree_feature_indexes_frequency, benign_katz_feature_indexes_frequency)
    union34_benign = np.union1d(benign_closeness_feature_indexes_frequency, benign_harmonic_feature_indexes_frequency)
    union1234_benign = np.union1d(union12_benign, union34_benign)
    print("Union of the two arrays:", union1234_benign.shape)

    # # 计算并集 benign
    union12 = np.union1d(malware_degree_feature_indexes_frequency, malware_katz_feature_indexes_frequency)
    union34 = np.union1d(malware_closeness_feature_indexes_frequency, malware_harmonic_feature_indexes_frequency)
    union1234_malware = np.union1d(union12, union34)
    print("Union of the two arrays:", union1234_malware.shape)
    #
    union_benign_malware = np.union1d(union1234_malware, union1234_benign)
    print("Union of the two arrays:", union_benign_malware.shape)

    # #找出敏感api, 保存到文件
    sensitive_apis = obtain_sensitive_apis()
    # 所有的基于frequency的重要特征，包括d
    with open('new_important_sensitive_apis.txt', 'w') as file:
        for i in union_benign_malware:
            line = sensitive_apis[i]
            file.write(line + '\n')  # 在每个字符串后添加换行符

    print('done!')
    #
    with open('malware_important_sensitive_apis.txt', 'w') as file:
        for i in union1234_malware:
            line = sensitive_apis[i]
            file.write(line + '\n')  # 在每个字符串后添加换行符

    print('done!')

    with open('benign_important_sensitive_apis.txt', 'w') as file:
        for i in union1234_benign:
            line = sensitive_apis[i]
            file.write(line + '\n')  # 在每个字符串后添加换行符

    print('done!')

    #
    with open('benign_important_sensitive_apis_degree_sum.txt', 'w') as file:
        for i in benign_degree_feature_indexes_sum:
            line = sensitive_apis[i]
            file.write(line + '\n')
    #
    # print('malware_degree_feature_indexes_sum', len(malware_degree_feature_indexes_sum))
    with open('malware_important_sensitive_apis_degree_sum.txt', 'w') as file:
        for i in malware_degree_feature_indexes_sum:
            line = sensitive_apis[i]
            file.write(line + '\n')
    #
    print('benign_katz_feature_indexes_sum', len(benign_katz_feature_indexes_sum))
    with open('benign_important_sensitive_apis_katz_sum.txt', 'w') as file:
        for i in benign_katz_feature_indexes_sum:
            line = sensitive_apis[i]
            file.write(line + '\n')

    print('malware_katz_feature_indexes_sum', len(malware_katz_feature_indexes_sum))
    with open('malware_important_sensitive_apis_katz_sum.txt', 'w') as file:
        for i in malware_katz_feature_indexes_sum:
            line = sensitive_apis[i]
            file.write(line + '\n')

    print('benign_harmonic_feature_indexes_sum', len(benign_harmonic_feature_indexes_sum))
    with open('benign_important_sensitive_apis_harmonic_sum.txt', 'w') as file:
        for i in benign_harmonic_feature_indexes_sum:
            line = sensitive_apis[i]
            file.write(line + '\n')

    print('malware_harmonic_feature_indexes_sum', len(malware_harmonic_feature_indexes_sum))
    with open('malware_important_sensitive_apis_harmonic_sum.txt', 'w') as file:
        for i in malware_harmonic_feature_indexes_sum:
            line = sensitive_apis[i]
            file.write(line + '\n')

    print('benign_closeness_feature_indexes_sum', len(benign_closeness_feature_indexes_sum))
    with open('benign_important_sensitive_apis_closeness_sum.txt', 'w') as file:
        for i in benign_closeness_feature_indexes_sum:
            line = sensitive_apis[i]
            file.write(line + '\n')

    print('malware_closeness_feature_indexes_sum', len(malware_closeness_feature_indexes_sum))
    with open('malware_important_sensitive_apis_closeness_sum.txt', 'w') as file:
        for i in malware_closeness_feature_indexes_sum:
            line = sensitive_apis[i]
            file.write(line + '\n')









    # # 2018_malware
    # X_malware_2018, Y_malware_2018 = dataset_one_feature('degree', '2018', '1')
    # # 2019_malware
    # # X_malware_2019, Y_malware_2019 = dataset_one_feature('degree', '2019', '1')
    # # 2022_malware
    # X_malware_2022, Y_malware_2022 = dataset_one_feature('degree', '2022', '1')
    # # 2023_malware
    # X_malware_2023, Y_malware_2023 = dataset_one_feature('degree', '2023', '1')
    #
    # # 合并
    # X_malware = X_malware_2018 + X_malware_2022 + X_malware_2023
    # X_malware = np.array(X_malware)
    # print(X_malware.shape)
    #
    # malware_feature_indexes = find_important_features(X_malware)
    # print("degree")
    # print(malware_feature_indexes)
    #
    # #比较benign和malware的重要特征
    # print("benign_feature_indexes")
    # cnt = 0
    # for i in benign_feature_indexes:
    #     if i not in malware_feature_indexes:
    #         cnt = cnt + 1

    # print(cnt)


class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.conv4 = GCNConv(64, 128)
        self.conv5 = GCNConv(128, 128)
        self.fc = nn.Linear(128, num_classes)  # 适配二分类输出

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_index = edge_index.t()
        # 通过五层 GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # 全局平均池化来聚合图的特征
        x = global_mean_pool(x, batch)  # 使用batch信息进行池化

        # 通过一个全连接层来进行分类
        x = self.fc(x)

        # 对于二分类，最后使用sigmoid激活函数（如果使用BCEWithLogitsLoss则不需要显式sigmoid）
        # return torch.sigmoid(x)  # 或者直接返回x，如果你使用的是BCEWithLogitsLoss损失函数
        return x

def obtain_gcn_feature(file, label):
    fcg = FCG(file, label, 0, True)
    # fcg.cal_centralities()
    # print("fcg nodes", len(fcg.nodes))
    # print("fcg edges", len(fcg.edges))
    edge_index = []
    degree_features = []
    for edge in fcg.edges:
        edge_index.append([edge[0], edge[1]])
    for node in fcg.nodes:
        in_degree = fcg.current_call_graph.in_degree(node)
        out_degree = fcg.current_call_graph.out_degree(node)
        degree_features.append([in_degree, out_degree])
    degree_features = torch.tensor(degree_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor(label, dtype=torch.long)
    data = Data(x=degree_features, edge_index=edge_index, y=y)
    return data

def gcn_test(data_list):
    # print("gcn_test")
    # data_list = []
    # MAX_READ_FILES = 600
    # benign_dir = "/data/b/shiwensong/dataset"
    # benign_dir_year = ["androzoo_benign_2018_gexf",
    #                              "androzoo_benign_2019_gexf",
    #                              "androzoo_benign_2020_gexf",
    #                              "androzoo_benign_2021_gexf",
    #                              "androzoo_benign_2022_gexf",
    #                              "androzoo_benign_2023_gexf"]
    # 
    # for year in benign_dir_year:
    #     benign_gexf_filepath = os.path.join(benign_dir, year)
    #     # 遍历benign_gexf_filepath，选择前100个文件
    #     benign_gexf_files = os.listdir(benign_gexf_filepath)
    #     benign_gexf_files = [os.path.join(benign_gexf_filepath, file) for file in benign_gexf_files[:MAX_READ_FILES]]
    #     print("benign_gexf_files", len(benign_gexf_files))
    # 
    #     pool = ThreadPool(50)
    #     print("pool", len(benign_gexf_files))
    #     data = pool.map(partial(obtain_gcn_feature, label=0), benign_gexf_files)
    #     data_list.append(data)
    # 
    # 
    # malware_dir = "/data/b/shiwensong/dataset"
    # malware_dir_year = ["virusshare2018_gexf",
    #                     "virusshare2019_gexf",
    #                     "virusshare2020_gexf",
    #                     "virusshare2021_gexf",
    #                     "virusshare2022_2_gexf",
    #                     "virusshare2023_3_gexf",]
    # malware_filenames_year = [
    #     "/data/b/shiwensong/dataset/feature_Nov30/dataset_malware_2018.csv",
    #     "/data/b/shiwensong/dataset/feature_Nov30/dataset_malware_2019.csv",
    #     "/data/b/shiwensong/dataset/feature_Nov30/dataset_malware_2020.csv",
    #     "/data/b/shiwensong/dataset/feature_Nov30/dataset_malware_2021.csv",
    #     "/data/b/shiwensong/dataset/feature_Nov30/dataset_malware_2022.csv",
    #     "/data/b/shiwensong/dataset/feature_Nov30/dataset_malware_2023.csv",
    # ]
    # 
    # for i in range(len(malware_dir_year)):
    #     malware_gexf_filepath = os.path.join(malware_dir, malware_dir_year[i])
    #     # 遍历benign_gexf_filepath，选择前100个文件
    #     print("malware_gexf_filepath", malware_gexf_filepath)
    # 
    #     df = pd.read_csv(malware_filenames_year[i])
    #     # get the first column
    #     sha256_list = df.iloc[:, 0]
    #     print(len(sha256_list.values))
    #     malware_file_path = []
    #     for sha256 in sha256_list.values:
    #         filename = sha256 + ".gexf"
    #         filename = os.path.join(malware_gexf_filepath, filename)
    #         print("malware filename", filename)
    #         malware_file_path.append(filename)
    # 
    #     pool = ThreadPool(50)
    #     print("pool", len(malware_file_path))
    #     data = pool.map(partial(obtain_gcn_feature, label=1), malware_file_path)
    #     data_list.append(data)


    loader = DataLoader(data_list, batch_size=1, shuffle=True)
    # 假设每个节点有2个特征，且问题是一个二分类问题
    model = GCN(num_node_features=2, num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    # 放到GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for epoch in range(30):  # 训练200个epoch
        for data in loader:  # 遍历DataLoader中的每个batch
            data = data.to(device)
            optimizer.zero_grad()  # 清除梯度
            out = model(data)  # 前向传播
            # loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 计算损失
            loss = criterion(out, data.y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    # 保存模型
    torch.save(model.state_dict(), f'gcn_model.pth')
    # 测试模型
    model.eval()
    correct = 0
    for data in loader:  # test_loader是你的测试数据加载器
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)  # 使用softmax的最大值作为预测
        correct += pred.eq(data.y).sum().item()  # 累加正确预测的数量
    print(f'Accuracy: {correct / len(loader.dataset)}')

def benign_dataset_mamadroid_feature(type = 'family'):
    X_benign = csr_matrix((0, 121))
    Y_benign = []
    for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
        feature_dir = f'/data/b/shiwensong/dataset/feature_Nov30/mamadroid_{type}_0_{year}.pkl'
        X_train, Y_train = obtain_mamadroid_feature(feature_dir)
        #取80%
        length = int(len(Y_train) * 0.8)
        X_train = X_train[:length]
        Y_train = Y_train[:length]
        X_benign = vstack([X_benign, X_train])
        Y_benign.extend(Y_train)
    print("X_benign", X_benign.shape)
    print("Y_train", len(Y_benign))
    return X_benign, Y_benign

def benign_dataset_mamadroid_feature_for_shap(type = 'family'):
    X_benign = csr_matrix((0, 121))
    Y_benign = []
    for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
        feature_dir = f'/data/b/shiwensong/dataset/feature_Nov30/mamadroid_{type}_0_{year}.pkl'
        X_train, Y_train = obtain_mamadroid_feature(feature_dir)
        #random selection
        indices = np.random.choice(X_train.shape[0], 100, replace=False)
        X_train = X_train[indices]
        Y_train = Y_train[:100]
        X_benign = vstack([X_benign, X_train])
        Y_benign.extend(Y_train)
    print("X_benign", X_benign.shape)
    print("Y_train", len(Y_benign))
    return X_benign, Y_benign

def test_benign_dataset_mamadroid_feature(type = 'family'):
    X_benign = csr_matrix((0, 121))
    Y_benign = []
    for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
        feature_dir = f'/data/b/shiwensong/dataset/feature_Nov30/mamadroid_{type}_0_{year}.pkl'
        X_train, Y_train = obtain_mamadroid_feature(feature_dir)
        #取80%
        length_80 = int(len(Y_train) * 0.8)
        length_90 = int(len(Y_train) * 0.9)
        X_train = X_train[length_80:length_90]
        Y_train = Y_train[length_80:length_90]
        X_benign = vstack([X_benign, X_train])
        Y_benign.extend(Y_train)
    print("X_benign", X_benign.shape)
    print("Y_train", len(Y_benign))
    return X_benign, Y_benign

def val_benign_dataset_mamadroid_feature(type = 'family'):
    X_benign = csr_matrix((0, 121))
    Y_benign = []
    for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
        feature_dir = f'/data/b/shiwensong/dataset/feature_Nov30/mamadroid_{type}_0_{year}.pkl'
        X_train, Y_train = obtain_mamadroid_feature(feature_dir)
        #取80%
        length = int(len(Y_train) * 0.9)
        X_train = X_train[length:]
        Y_train = Y_train[length:]
        X_benign = vstack([X_benign, X_train])
        Y_benign.extend(Y_train)
    print("X_benign", X_benign.shape)
    print("Y_train", len(Y_benign))
    return X_benign, Y_benign

def malware_dataset_mamadroid_feature(type = 'family'):
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    X_malware = csr_matrix((0, 121))
    Y_malware = []
    for i in range(len(years)):
        year = years[i]
        feature_dir = f'/data/b/shiwensong/dataset/feature_Nov30/mamadroid_{type}_1_{year}.pkl'
        X_train, Y_train = obtain_mamadroid_feature(feature_dir)
        length = int(len(Y_train) * 0.8)
        X_train = X_train[:length]
        Y_train = Y_train[:length]
        X_malware = vstack([X_malware, X_train])
        Y_malware.extend(Y_train)

    print("X_malware", X_malware.shape)
    print("Y_malware", len(Y_malware))
    return X_malware, Y_malware

def malware_dataset_mamadroid_feature_for_shap(type = 'family'):
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    X_malware = csr_matrix((0, 121))
    Y_malware = []
    for i in range(len(years)):
        year = years[i]
        feature_dir = f'/data/b/shiwensong/dataset/feature_Nov30/mamadroid_{type}_1_{year}.pkl'
        X_train, Y_train = obtain_mamadroid_feature(feature_dir)
        indices = np.random.choice(X_train.shape[0], 100, replace=False)
        X_train = X_train[indices]
        Y_train = Y_train[:100]
        X_malware = vstack([X_malware, X_train])
        Y_malware.extend(Y_train)

    print("X_malware", X_malware.shape)
    print("Y_malware", len(Y_malware))
    return X_malware, Y_malware

def test_malware_dataset_mamadroid_feature(type = 'family'):
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    X_malware = csr_matrix((0, 121))
    Y_malware = []
    for i in range(len(years)):
        year = years[i]
        feature_dir = f'/data/b/shiwensong/dataset/feature_Nov30/mamadroid_{type}_1_{year}.pkl'
        X_train, Y_train = obtain_mamadroid_feature(feature_dir)
        length = int(len(Y_train) * 0.8)
        X_train = X_train[length:]
        Y_train = Y_train[length:]
        X_malware = vstack([X_malware, X_train])
        Y_malware.extend(Y_train)

    print("X_malware", X_malware.shape)
    print("Y_malware", len(Y_malware))
    return X_malware, Y_malware

def obtain_mamadroid_feature(file_path):
    # 加载 .pkl 文件
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    # 解包数据
    sha256, X_train, labels = loaded_data

    # 初始化筛选用的集合和索引列表
    sample_count = set()
    selected_indices = []

    for i, sha in enumerate(sha256):
        if len(sample_count) >= 1000:
            break
        if sha not in sample_count:
            sample_count.add(sha)
            selected_indices.append(i)

    # 使用选中的索引从csr_matrix中选择行
    final_X_train = X_train[selected_indices]
    final_labels = np.array(labels)[selected_indices]

    return final_X_train, final_labels

def cal_and_test_model(X_test, Y_test, model, type = None):
    res = model.predict(X_test)
    malware_error = 0
    malware_right = 0
    benign_error = 0
    benign_right = 0
    if type is not None:
        print("res", res)
        for i in range(60):
            # print(res[i], Y_test[i])
            if Y_test[i] == 0:
                if res[i][0] < 0:
                    benign_right += 1
                else:
                    benign_error += 1

            else:
                if res[i][0] > 0:
                    malware_right += 1
                else:
                    malware_error += 1

    else:
        for i in range(60):
            # print(res[i], Y_test[i])
            if Y_test[i] == 0:
                if res[i] == 0:
                    benign_right += 1
                else:
                    benign_error += 1

            else:
                if res[i] == 1:
                    malware_right += 1
                else:
                    malware_error += 1

    print("malware_right", malware_right)
    print("malware_error", malware_error)
    print("benign_right", benign_right)
    print("benign_error", benign_error)
    return [benign_right, benign_error, malware_right, malware_error]

def cal_and_test_substitute_model(X_test, ori_model, sub_model):
    ori_res = ori_model.predict(X_test)
    sub_res = sub_model.predict(X_test)
    malware_error = 0
    malware_right = 0
    benign_error = 0
    benign_right = 0
    for i in range(len(ori_res)):
        # print(res[i], Y_test[i])
        if ori_res[i] == 0:
            if sub_res[i][0] < 0:
                benign_right += 1
            else:
                benign_error += 1

        else:
            if sub_res[i][0] > 0:
                malware_right += 1
            else:
                malware_error += 1


    print("malware_right", malware_right)
    print("malware_error", malware_error)
    print("benign_right", benign_right)
    print("benign_error", benign_error)
    
# 函数来合并边和节点特征列表
def merge_data(edge_list, node_features_list):
    cumulative_nodes = 0
    edge_index_list = []
    x_list = []
    batch_list = []

    for i, (edges, features) in enumerate(zip(edge_list, node_features_list)):
        num_nodes = features.size(0)
        x_list.append(features)
        batch_list.append(torch.full((num_nodes,), i, dtype=torch.long))
        # 偏移边的节点索引
        edge_index_list.append(edges + cumulative_nodes)
        cumulative_nodes += num_nodes

    return torch.cat(edge_index_list, dim=1), torch.cat(x_list, dim=0), torch.cat(batch_list, dim=0)

def train_and_save_model(edge_data, feature_data, labels):
    model_save_path = 'allfeatures_6yeardataset_1000samples/'

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 初始化模型和训练过程
    model = GNN(num_layers=5, in_channels=2, hidden_channels=32, out_channels=1, dropout_ratio=0.2,
                gnn_type='gcn', JK='last').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    min_loss = float('inf')
    zero_loss_count = 0
    max_zero_loss_threshold = 10

    # 训练模型
    model.train()
    for epoch in range(200):  # 训练100个epoch
        for edges, features, label in zip(edge_data, feature_data, labels):
            edge_index = torch.cat([edges], dim=1).to(device)
            x = features.reshape(-1, 2).to(device)
            y = torch.tensor(label, dtype=torch.long).to(device)
            data = Data(x=x, edge_index=edge_index, y=y)

            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')

        if loss.item() == 0:
            zero_loss_count += 1
        else:
            zero_loss_count = 0

        if zero_loss_count >= max_zero_loss_threshold:
            print("Early stopping triggered due to zero loss.")
            break

        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(model_save_path, 'baseline_5layers_GCN.pth'))
            print(f"Model saved at epoch {epoch} with loss {min_loss}.")



                
# def test_gcn_model(test_edge_data, test_feature_data, test_labels, model_path):
#     model = GNN(num_layers=5, in_channels=2, hidden_channels=32, out_channels=2, dropout_ratio=0.2, gnn_type='gcn', JK='last')
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#
#     correct = 0
#     total = 0
#     criterion = torch.nn.CrossEntropyLoss()
#
#     with torch.no_grad():
#         for edges, features, label in zip(test_edge_data, test_feature_data, test_labels):
#             edge_index = torch.cat([edges], dim=1)
#             x = features.reshape(-1, 1)
#             y = torch.tensor(label, dtype=torch.long)
#
#             data = Data(x=x, edge_index=edge_index, y=y)
#
#             output = model(data.x, data.edge_index, None)
#             loss = criterion(output, data.y)
#             predicted = output.argmax(dim=1)
#             total += data.y.size(0)
#             print("predict", predicted)
#             correct += (predicted == data.y).sum().item()
#
#     accuracy = correct / total
#     print(f'Test Accuracy: {accuracy * 100:.2f}%')
#     return accuracy
def test_gcn_model(test_edge_data, test_feature_data, test_labels, model_path):
    # Initialize the model with the same configuration as during training
    model = GNN(num_layers=5, in_channels=2, hidden_channels=32, out_channels=2, dropout_ratio=0.2, gnn_type='gcn', JK='last')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():  # Operations inside don't track history
        for edges, features, label in zip(test_edge_data, test_feature_data, test_labels):
            # Process edges and features to match the input requirements of the model
            edge_index = torch.cat([edges], dim=1)  # Assuming edges is already a tensor
            x = features.reshape(-1, 2)  # Ensuring features are in the correct shape
            y = torch.tensor(label, dtype=torch.long)  # Ensure labels are in the correct shape and type

            # Create a data object as done in training
            data = Data(x=x, edge_index=edge_index, y=y)

            # Run the model on the test data
            output = model(data.x, data.edge_index)  # Batch parameter is not needed for single graph
            print('output', output)
            res = model.predict(data.x, data.edge_index)
            print(res)
            loss = criterion(output, data.y)  # Calculate loss for monitoring
            predicted = output.argmax(dim=1)  # Get the class with the highest probability
            total += data.y.size(0)  # Count total samples
            correct += (predicted == data.y).sum().item()  # Count correct predictions

            # Optionally print predictions for each sample
            print("Predicted:", predicted, "Actual:", data.y)

    # Calculate the accuracy
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy


def load_data():
    # 打开文件并加载数据
    with open('baseline_malware_dataset.pkl', 'rb') as f:
        edge_malware, node_features_malware, malware_y_pre_labels = pickle.load(f)

    with open('baseline_benign_dataset.pkl', 'rb') as f:
        edge_benign, node_features_benign, benign_y_pre_labels = pickle.load(f)

    # 创建标签数组
    labels = np.concatenate([np.zeros(len(benign_y_pre_labels)), np.ones(len(malware_y_pre_labels))])

    # 初始化数据列表
    data_list = []

    # 遍历恶意软件样本
    for i in range(len(edge_malware)):
        y = torch.tensor([1], dtype=torch.long)  # 恶意软件标记为 1
        x = torch.tensor(node_features_malware[i], dtype=torch.float)
        edge_index = torch.tensor(edge_malware[i], dtype=torch.long).t().contiguous() if edge_malware[i].shape[0] == 2 else torch.tensor(edge_malware[i], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    # 遍历良性软件样本
    for i in range(len(edge_benign)):
        y = torch.tensor([0], dtype=torch.long)  # 良性软件标记为 0
        x = torch.tensor(node_features_benign[i], dtype=torch.float)
        edge_index = torch.tensor(edge_benign[i], dtype=torch.long).t().contiguous() if edge_benign[i].shape[0] == 2 else torch.tensor(edge_benign[i], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list



def obtain_ae_train_dataset(feature_type, length = 10):
    if feature_type == 'family':
        feature_type = 'mamadroid'

    file_path = f'adversarial_{feature_type}_MLP.csv'
    #read the csv file
    df = pd.read_csv(file_path)
    # 获取前length行
    df_subset = df.head(length)
    return df_subset

def obtain_ae_test_dataset(feature_type, length = 60):
    if feature_type == 'family':
        feature_type = 'mamadroid'

    file_path = f'adversarial_{feature_type}_MLP_withoutshap.csv'
    #read the csv file
    df = pd.read_csv(file_path)
    # 获取前length行
    df_subset = df.head(length)
    return df_subset


if __name__ == '__main__':
    # gcn_test()
    # X_benign, Y_benign = benign_dataset_avg_feature()
    # X_malware, Y_malware = malware_dataset_avg_feature()
    # feature_type = ''
    #
    feature_types = ['degree', 'katz', 'closeness', 'harmonic', 'avg', 'all']
    # # #
    model_output_path = './430features_6yearsdataset_1000samples/'
    logger = init_logger(10, '/data/c/shiwensong/project/malwareGA/log/', 'malscan')

    # feature_type = 'degree'
    # path = f'{model_output_path}{feature_type}MLP.pth'
    # print(path)

    # train_dataset_dir = '/data/b/guoqi/icse25/malwareGA/dataset/trainset_apigraph50_features.pkl'
    # train_dataset = pd.read_pickle(train_dataset_dir)
    # # obtain the first
    # benign = [row[0] for row in train_dataset[:6000]]
    # benign_labels = [0] * 6000
    # 
    # 
    # malware = [row[0] for row in train_dataset[6000:12000]]
    # malware_labels = [1] * 6000
    # 
    # #垂直拼接
    # benign.extend(malware)
    # X_dataset = np.array(benign)
    # benign_labels.extend(malware_labels)
    # Y_dataset = np.array(benign_labels)
    # train_model(benign, benign_labels, 'KNN_10', model_output_path='./430features_6yearsdataset_1000samples/', feature_type = 'apigraph_benign')
    # train_model(malware, malware_labels, 'KNN_10', model_output_path='./430features_6yearsdataset_1000samples/', feature_type = 'apigraph_malware')
    # knn_1 = joblib.load(f'/data/b/guoqi/icse25/malwareGA/model/apigraph/apigraphknn_1.pkl')
    # train_substitute_Model(X_dataset, knn_1, 'MLP', 'knn_1', model_output_path, 'apigraph')
    # knn_3 = joblib.load(f'/data/b/guoqi/icse25/malwareGA/model/apigraph/apigraphknn_3.pkl')
    # train_substitute_Model(X_dataset, knn_3, 'MLP', 'knn_3', model_output_path, 'apigraph')

    # 假设 obtain_gcn_malware_dataset 和 obtain_gcn_benign_dataset 已定义并正确返回每个样本的边索引和节点特征
    # edge_malware, node_features_malware = obtain_gcn_malware_dataset()
    # edge_benign, node_features_benign = obtain_gcn_benign_dataset()
    #
    # # 初始化早停变量
    # min_loss = float('inf')
    # zero_loss_count = 0
    # max_zero_loss_threshold = 5  # 连续为零损失的次数阈值
    #
    # # 处理每个样本
    # for edges, features, label in zip(edge_benign + edge_malware, node_features_benign + node_features_malware,
    #                                   [0] * len(node_features_benign) + [1] * len(node_features_malware)):
    #     edge_index = torch.cat([edges], dim=1)  # 假设 edges 已是 tensor
    #     x = features.reshape(-1, 1)  # 假设 features 已是 tensor
    #     y = torch.tensor([label], dtype=torch.long)
    #
    #     # 创建Data对象
    #     data = Data(x=x, edge_index=edge_index, y=y)
    #
    #     print(x)
    #
    #     # 初始化模型和训练过程
    #     model = GNN(num_layers=3, in_channels=x.size(1), hidden_channels=32, out_channels=2, dropout_ratio=0.2,
    #                 gnn_type='gcn', JK='last')
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #     criterion = torch.nn.CrossEntropyLoss()
    #
    #     # 训练模型
    #     model.train()
    #     for epoch in range(100):  # 训练100个epoch
    #         optimizer.zero_grad()
    #         output = model(data.x, data.edge_index, None)  # 对于单个图，不需要 batch 参数
    #         loss = criterion(output, data.y)
    #         loss.backward()
    #         optimizer.step()
    #         print(f'Epoch: {epoch}, Loss: {loss.item()}')

    #调用函数
    # model = joblib.load(f'{model_output_path}degree3Layers_RandomForest.pkl')
    # edge_malware, node_features_malware, y_pre_labels_malware = obtain_gcn_malware_dataset(None, 'in_out_degree')
    # print("y_pre_labels_malware", y_pre_labels_malware)
    # edge_benign, node_features_benign, y_pre_labels_benign = obtain_gcn_benign_dataset(None, 'in_out_degree')
    # print("y_pre_labels_benign", y_pre_labels_benign)
    # y_pre_labels_malware = np.array(y_pre_labels_malware)
    # y_pre_labels_benign = np.array(y_pre_labels_benign)
    # labels = np.hstack((y_pre_labels_benign, y_pre_labels_malware))
    # print("labels", labels.shape)


    # 打开文件并加载数据
    # with open('baseline_malware_dataset.pkl', 'rb') as f:
    #     edge_malware, node_features_malware, malware_y_pre_labels = pickle.load(f)
    #
    # with open('baseline_benign_dataset.pkl', 'rb') as f:
    #     edge_benign, node_features_benign, benign_y_pre_labels = pickle.load(f)
    #
    # labels = np.vstack([np.array(len(benign_y_pre_labels) * [0]),
    #                     np.array(len(malware_y_pre_labels) * [1])]).reshape(-1, 1)

    # data_list = load_data()
    # gcn_test(data_list)
    # train_and_save_model(edge_benign + edge_malware, node_features_benign + node_features_malware, labels)

    # labels = np.vstack([np.array(len(malware_y_pre_labels) * [1]),
    #                     np.array(len(benign_y_pre_labels) * [0])]).reshape(-1, 1)

    # labels = np.array(len(benign_y_pre_labels) * [0]).reshape(-1, 1)
    # labels = np.vstack((malware_y_pre_labels, benign_y_pre_labels))

    # print(len(edge_malware))
    # print(len(edge_benign))
    # print(labels.shape)

    # test_gcn_model(edge_benign + edge_malware, node_features_benign + node_features_malware, labels, './allfeatures_6yeardataset_1000samples/baseline_5layers_GCN.pth')
    # test_gcn_model(edge_benign, node_features_benign, labels, './allfeatures_6yeardataset_1000samples/baseline_5layers_GCN.pth')


    #try gcn
    # gcn_test()
    #
    # for type in ['apigraph']:
    #     print("feature_type", type)
    #     X_dataset = None
    #     Y_dataset = None
    #     if type == 'family':
    #         X_benign, Y_benign = benign_dataset_mamadroid_feature(type)
    #         X_malware, Y_malware = malware_dataset_mamadroid_feature(type)
    #
    #     elif type == 'avg':
    #         X_benign, Y_benign = benign_dataset_avg_feature()
    #         X_malware, Y_malware = malware_dataset_avg_feature()
    #     elif type == 'all':
    #         X_benign, Y_benign = benign_dataset()
    #         X_malware, Y_malware = malware_dataset()
    #     elif type == 'apigraph':
    #         train_dataset_dir = '/data/b/guoqi/icse25/malwareGA/dataset/trainset_apigraph50_features.pkl'
    #         train_dataset = pd.read_pickle(train_dataset_dir)
    #         # obtain the first
    #         benign = [row[0] for row in train_dataset[:6000]]
    #         benign_labels = [0] * 6000
    #
    #         malware = [row[0] for row in train_dataset[6000:12000]]
    #         malware_labels = [1] * 6000
    #
    #         # 垂直拼接
    #         benign.extend(malware)
    #         X_dataset = np.array(benign)
    #         benign_labels.extend(malware_labels)
    #         Y_dataset = np.array(benign_labels)
    #     else:
    #         X_benign, Y_benign = benign_dataset_1feature(type)
    #         X_malware, Y_malware = malware_dataset_1feature(type)
    #     # X_dataset = np.vstack((X_benign, X_malware))
    #     # Y_dataset = np.hstack((Y_benign, Y_malware))
    #     if type != 'apigraph':
    #         X_dataset = vstack([X_benign, X_malware])
    #         Y_dataset = np.hstack((Y_benign, Y_malware))
    #     print(X_dataset.shape)
    #     print(Y_dataset.shape)
    #
    #     #
    #     lengths = [10, 30, 50, 70, 90]
    #     model_output_path = 'retrain_models/'
    #     for length in lengths:
    #         res = obtain_ae_train_dataset(type, length)
    #         print("res", res.shape)
    #         res = np.array(res)
    #         X_dataset = vstack([X_dataset, res])
    #         Y_dataset = np.hstack((Y_dataset, np.array([1] * length)))
    #
    #         new_type = type + str(length)
    #         # model3 = train_model(X_dataset, Y_dataset, 'KNN_1', model_output_path=model_output_path, type = new_type)
    #         # model4 = train_model(X_dataset, Y_dataset, 'KNN_3', model_output_path=model_output_path, type = new_type)
    #         model5 = train_model(X_dataset, Y_dataset, 'MLP', model_output_path=model_output_path, type = new_type)
    #         # model6 = train_model(X_dataset, Y_dataset, 'RandomForest', model_output_path=model_output_path, type = new_type)
    #         # model7 = train_model(X_dataset, Y_dataset, 'AdaBoost', model_output_path=model_output_path, type = new_type)
    # #
    # sys.exit(0)
        # save as pkl, format : feature, label
        # with open(f'/data/b/shiwensong/dataset/feature_Nov30/mamadroid_family.pkl', 'wb') as f:
        #     pickle.dump([X_dataset, Y_dataset], f)
        #
        # print("feature_type", type)
        # model6 = train_model(X_dataset, Y_dataset, 'RandomForest', model_output_path='./allfeatures_6yeardataset_1000samples/', type = type + 'Default_')


    for cur_type in ['apigraph']:

        # if type == 'avg':
        #     X_benign_test, Y_benign_test = test_benign_dataset_avg_feature()
        #     X_malware_test, Y_malware_test = test_malware_dataset_avg_feature()
        # elif type == 'all':
        #     X_benign_test, Y_benign_test = test_benign_dataset()
        #     X_malware_test, Y_malware_test = test_malware_dataset()
        # elif type == 'family':
        #     X_benign_test, Y_benign_test = test_benign_dataset_mamadroid_feature(type)
        # else:
        #     X_benign_test, Y_benign_test = test_benign_dataset_1feature(type)
        #     X_malware_test, Y_malware_test = test_malware_dataset_1feature(type)

        # X_test_ori = np.vstack((X_benign_test, X_malware_test))
        # Y_test_ori = np.hstack((Y_benign_test, Y_malware_test))

        lengths = [10, 30, 50, 70, 90]
        # lengths = [10]
        # model_output_path = './430features_6yearsdataset_1000samples/'
        model_output_path = 'retrain_models/'
        test_results = []
        for length in lengths:
            if cur_type == 'katz' and length == 90:
                continue


            print("type", cur_type)
            res = obtain_ae_test_dataset(cur_type)
            res = np.array(res)

            print("length", length)
            print("res", res.shape)
            print(cur_type)
            X_test_ae = res
            len = 60
            Y_test_ae = np.array([1] * len)

            # model_path1 = f'{model_output_path}{cur_type}{str(length)}knn_1.pkl'
            # model1 = load_test_model(model_path1)
            # res1 = cal_and_test_model(X_test_ae, Y_test_ae, model1)
            # test_results.append(res1)
            #
            # model_path2 = f'{model_output_path}{cur_type}{str(length)}knn_3.pkl'
            # model2 = load_test_model(model_path2)
            # res2 = cal_and_test_model(X_test_ae, Y_test_ae, model2)
            # test_results.append(res2)
            #
            # model_path3 = f'{model_output_path}{cur_type}{str(length)}RandomForest.pkl'
            # model3 = load_test_model(model_path3)
            # res3 = cal_and_test_model(X_test_ae, Y_test_ae, model3)
            # test_results.append(res3)
            #
            # model_path4 = f'{model_output_path}{cur_type}{str(length)}AdaBoost.pkl'
            # model4 = load_test_model(model_path4)
            # res4 = cal_and_test_model(X_test_ae, Y_test_ae, model4)
            # test_results.append(res4)
            #
            model_path5 = f'{model_output_path}{cur_type}{str(length)}MLP.h5'
            # model_path5 = '/data/b/guoqi/icse25/malwareGA/model/apigraph/apigraphMLP.h5'
            shape = 430
            if cur_type == 'all':
                shape = 1720
            elif cur_type == 'family':
                shape = 121
            elif cur_type == 'apigraph':
                shape = 2704
            model5 = load_test_model(model_path5, shape)
            res5 = cal_and_test_model(X_test_ae, Y_test_ae, model5, 'mlp')
            test_results.append(res5)

            test_results.append([0,0,0,length])

        #
        # print("type", cur_type)
        # res = obtain_ae_test_dataset(cur_type)
        # res = np.array(res)
        #
        # print("res", res.shape)
        # print(cur_type)
        # X_test_ae = res
        # len = 60
        # Y_test_ae = np.array([1] * len)

        # model_path1 = f'{model_output_path}{cur_type}knn_1.pkl'
        # model1 = load_test_model(model_path1)
        # res1 = cal_and_test_model(X_test_ae, Y_test_ae, model1)
        # test_results.append(res1)
        #
        # model_path2 = f'{model_output_path}{cur_type}knn_3.pkl'
        # model2 = load_test_model(model_path2)
        # res2 = cal_and_test_model(X_test_ae, Y_test_ae, model2)
        # test_results.append(res2)
        #
        # model_path3 = f'{model_output_path}{cur_type}3Layers_RandomForest.pkl'
        # model3 = load_test_model(model_path3)
        # res3 = cal_and_test_model(X_test_ae, Y_test_ae, model3)
        # test_results.append(res3)
        #
        # model_path4 = f'{model_output_path}{cur_type}100Estimator_AdaBoost.pkl'
        # model4 = load_test_model(model_path4)
        # res4 = cal_and_test_model(X_test_ae, Y_test_ae, model4)
        # test_results.append(res4)
        #
        # model_path5 = f'{model_output_path}{cur_type}MLP.h5'
        # # model_path5 = '/data/b/guoqi/icse25/malwareGA/model/apigraph/apigraphMLP.h5'
        # shape = 430
        # if cur_type == 'all':
        #     shape = 1720
        # elif cur_type == 'family':
        #     shape = 121
        # elif cur_type == 'apigraph':
        #     shape = 2704
        # model5 = load_test_model(model_path5, shape)
        # res5 = cal_and_test_model(X_test_ae, Y_test_ae, model5, 'mlp')
        # test_results.append(res5)

        # test_results.append([0,0,0,length])
        #
        # #save as csv
        df = pd.DataFrame(test_results, columns=['benign_right', 'benign_error', 'malware_right', 'malware_error'])
        df.to_csv(f'retrain_models/retrain_test_results/{cur_type}_test.csv', index=False)
    #
    #
    #     #val
    #     X_benign_val = None
    #     X_malware_val = None
    #     if type == 'avg':
    #         X_benign_val, Y_benign_val = val_benign_dataset_avg_feature()
    #         X_malware_val, Y_malware_val = val_malware_dataset_avg_feature()
    #     elif type == 'all':
    #         X_benign_val, Y_benign_val = val_benign_dataset()
    #         X_malware_val, Y_malware_val = val_malware_dataset()
    #     elif type == 'family':
    #         X_benign_val, Y_benign_val = val_benign_dataset_mamadroid_feature(type)
    #     else:
    #         X_benign_val, Y_benign_val = val_benign_dataset_1feature(type)
    #         X_malware_val, Y_malware_val = val_malware_dataset_1feature(type)
    #
    #
    #
    #     target_models = ['100Estimator_AdaBoost', '3Layers_RandomForest', 'knn_1', 'knn_3']
    #
    #     for target_model in target_models:
    #         knn_1 = joblib.load(f'{output_model_dir}/{type}{target_model}.pkl')
    #         y_pred_train = knn_1.predict(X_dataset)
    #
    #         train_data = SimpleDataset(X_dataset, y_pred_train)
    #         substitute_model = MLP(in_channels=X_dataset.shape[1],
    #                                hidden_channels=1024,
    #                                out_channels=2,
    #                                attention=False)
    #
    #         X_test = np.vstack((X_benign_test, X_malware_test))
    #         y_pred_test = knn_1.predict(X_test)
    #
    #         test_data = SimpleDataset(X_test, y_pred_test)
    #         print("X_test", X_test.shape)
    #
    #         X_val = np.vstack((X_benign_val, X_malware_val))
    #         y_pred_val = knn_1.predict(X_val)
    #
    #         val_data = SimpleDataset(X_val, y_pred_val)
    #         print("val_data", X_val.shape)
    #
    #
    #         # transformed_train_data = pca.transform(train_data)
    #         # transformed_val_data = pca.transform(test_data)
    #
    #
    #         mlp_train(model=substitute_model,
    #                   logger=logger,
    #                   train_data=train_data,
    #                   val_data=val_data,
    #                   test_data=test_data,
    #                   model_path=f'{output_model_dir}/{type}_mlp_{target_model}.pth',
    #                   evaluation=True)

        #

        # 假设你已经拟合了一个PCA模型
        # pca = PCA(n_components=1720)
        # pca.fit(X_dataset)  # data 是你的训练数据
        # dump(pca, 'pca_model.joblib')


        # knn_1 = joblib.load(f'{model_output_path}{feature_type}knn_1.pkl')
        # train_substitute_Model(X_dataset, knn_1, 'MLP', 'knn_1', model_output_path, feature_type)
        # knn_3 = joblib.load(f'{model_output_path}{feature_type}knn_3.pkl')
        # train_substitute_Model(X_dataset, knn_3, 'MLP', 'knn_3', model_output_path, feature_type)

    #     Y_dataset = Y_dataset.reshape(-1, 1)
    #     print(Y_dataset.shape)
    #     # X_tensor = torch.tensor(X_dataset, dtype=torch.float32)
    #     # Y_tensor = torch.tensor(Y_dataset, dtype=torch.float32)
    #     # train_mlp_model(X_tensor, Y_tensor, 430, './430features_6yearsdataset_1000samples/', feature_type = feature_type + '430features_')
    #     # model1 = train_model(X_benign, Y_benign, 'KNN_10', model_output_path='./430features_6yearsdataset_1000samples/', feature_type = feature_type+'_benign')
    #     # model2 = train_model(X_malware, Y_malware, 'KNN_10', model_output_path='./430features_6yearsdataset_1000samples/', feature_type = feature_type +'_malware')
    #     model3 = train_model(X_dataset, Y_dataset, 'KNN_1', model_output_path='./allfeatures_6yeardataset_1000samples/', feature_type = feature_type)
    #     model4 = train_model(X_dataset, Y_dataset, 'KNN_3', model_output_path='./allfeatures_6yeardataset_1000samples/', feature_type = feature_type)
    #     model5 = train_model(X_dataset, Y_dataset, 'MLP', model_output_path='./allfeatures_6yeardataset_1000samples/', feature_type = feature_type)
    #     model6 = train_model(X_dataset, Y_dataset, 'RandomForest', model_output_path='./allfeatures_6yeardataset_1000samples/', feature_type = feature_type + 'Default_')
    #     model7 = train_model(X_dataset, Y_dataset, 'AdaBoost', model_output_path='./allfeatures_6yeardataset_1000samples/', feature_type = feature_type + '100Estimator_')

    # types = ['degree', 'katz', 'closeness', 'harmonic', 'avg', 'all']
    # model_name = ['knn_1', 'knn_3', 'MLP', 'RandomForest', 'AdaBoost']
    # # #
    # for feature_type in ['family']:
    #     if feature_type == 'family':
    #         X_benign, Y_benign = benign_dataset_mamadroid_feature(feature_type)
    #         X_malware, Y_malware = malware_dataset_mamadroid_feature(feature_type)
    #     elif feature_type == 'avg':
    #         X_benign, Y_benign = benign_dataset_avg_feature()
    #         X_malware, Y_malware = malware_dataset_avg_feature()
    #     elif feature_type == 'all':
    #         X_benign, Y_benign = benign_dataset()
    #         X_malware, Y_malware = malware_dataset()
    #     else:
    #         X_benign, Y_benign = benign_dataset_1feature(feature_type)
    #         X_malware, Y_malware = malware_dataset_1feature(feature_type)
    #     #only for family
    #     X_dataset = vstack([X_benign, X_malware])
    #     # X_dataset = np.vstack((X_benign, X_malware))
    #     Y_dataset = np.hstack((Y_benign, Y_malware))
    #     print("feature_type", feature_type)
    #     print(X_dataset.shape)
    #     print(Y_dataset.shape)
    #
    #     model_output_path = './430features_6yearsdataset_1000samples/'
    # # # #     # for cur_model_name in model_name:
    # # # #     #     if cur_model_name == 'KNN_10':
    # # # #     #         train_model(X_benign, Y_benign, cur_model_name, model_output_path, feature_type=feature_type+"_benign")
    # # # #     #         train_model(X_malware, Y_malware, cur_model_name, model_output_path, feature_type=feature_type+"_malware")
    # # # #     #     else:
    # #     train_model(X_dataset, Y_dataset, 'SVM', model_output_path, feature_type=feature_type)
    #     train_model(X_dataset, Y_dataset, 'RandomForest', model_output_path, feature_type=feature_type + '_three')
    #     train_model(X_dataset, Y_dataset, 'AdaBoost', model_output_path, feature_type=feature_type + '_standard')
    # #     #
    # #     # #for substitute model
    # #     # for cur_model_name in model_name:
    # #     #     if cur_model_name != 'KNN_10' and cur_model_name != 'MLP':
    # #     #         if cur_model_name == 'KNN_1' or cur_model_name == 'KNN_3':
    # #     #             cur_model_name = cur_model_name.lower()
    #     cur_model_name = '_standardRandomForest'
    #     model_path = f'{model_output_path}{feature_type}{cur_model_name}.pkl'
    #     model = joblib.load(model_path)
    #     train_substitute_Model_using_prob(X_dataset, model, 'MLP', cur_model_name, model_output_path, feature_type)
    # #
    # #
    # #     #
    #     ab_path = f'./430features_6yearsdataset_1000samples/{feature_type}SVM.pkl'
    #     ab = joblib.load(ab_path)
    #     train_substitute_Model(X_dataset, ab, 'MLP', 'SVM', './430features_6yearsdataset_1000samples/', feature_type = feature_type)
    # #
    #     rf_path = f'./430features_6yearsdataset_1000samples/{feature_type}_usx_newRandomForest.pkl'
    #     rf = joblib.load(rf_path)
    #     train_substitute_model_using_VAE(X_dataset, rf,  './430features_6yearsdataset_1000samples/')
    #     #
    #     # knn3_path = f'./430features_6yearsdataset_1000samples/{feature_type}knn_3.pkl'
    #     knn3 = joblib.load(knn3_path)
    #     train_substitute_Model(X_dataset, knn3, 'MLP', 'knn_3', './430features_6yearsdataset_1000samples/', feature_type)
    #
    #     knn1_path = f'./430features_6yearsdataset_1000samples/{feature_type}knn_1.pkl'
    #     knn1 = joblib.load(knn1_path)
    #     train_substitute_Model(X_dataset, knn1, 'MLP', 'knn_1', './430features_6yearsdataset_1000samples/', feature_type)
    #
    #     knn_1 = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}knn_1.pkl')
    #     MLP_knn_1 = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}MLP_knn_1.h5')
    #
    #     knn_3 = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}knn_3.pkl')
    #     MLP_knn_3 = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}MLP_knn_3.h5')
    #
    #     rf = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}RandomForest.pkl')
    #     MLP_rf = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}MLP_RandomForest.h5')
    #
    #     ab = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}AdaBoost.pkl')
    #     MLP_ab = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}MLP_AdaBoost.h5')


    # model_dir = './430features_6yearsdataset_1000samples/'
    # for feature_type in types:
    # #     print("feature_type", feature_type)
    # #     shape = 430
    #     if feature_type == 'avg':
    #         X_benign, Y_benign = test_benign_dataset_avg_feature()
    #         X_malware, Y_malware = test_malware_dataset_avg_feature()
    #     elif feature_type == 'all':
    #         X_benign, Y_benign = test_benign_dataset()
    #         X_malware, Y_malware = test_malware_dataset()
    #         shape = 1720
    #     elif feature_type == 'family':
    #         X_benign, Y_benign = test_benign_dataset_mamadroid_feature(feature_type)
    #         X_malware, Y_malware = test_malware_dataset_mamadroid_feature(feature_type)
    #         shape = 121
    #     else:
    #         X_benign, Y_benign = test_benign_dataset_1feature(feature_type)
    #         X_malware, Y_malware = test_malware_dataset_1feature(feature_type)
    #     X_dataset = vstack([X_benign, X_malware])
    #     Y_dataset = np.hstack((Y_benign, Y_malware))
    #     print(X_dataset.shape)
    #     print(Y_dataset.shape)
    #     print("feature_type", feature_type)
    #
    #     knn_1 = joblib.load(f'{model_dir}{feature_type}knn_1.pkl')
    #     model1 = load_test_model(f'{model_dir}{feature_type}MLP_knn_1.h5', shape)

    # train_dataset_dir = '/data/b/guoqi/icse25/malwareGA/dataset/trainset_apigraph50_features.pkl'
    # train_dataset = pd.read_pickle(train_dataset_dir)
    # # obtain the first
    # benign_2018 = [row[0] for row in train_dataset[800:1000]]
    # benign_2019 = [row[0] for row in train_dataset[1800:2000]]
    # benign_2020 = [row[0] for row in train_dataset[2800:3000]]
    # benign_2021 = [row[0] for row in train_dataset[3800:4000]]
    # benign_2022 = [row[0] for row in train_dataset[4800:5000]]
    # benign_2023 = [row[0] for row in train_dataset[5800:6000]]
    # benign_labels = [0] * 1200
    # benign_2018.extend(benign_2019)
    # benign_2018.extend(benign_2020)
    # benign_2018.extend(benign_2021)
    # benign_2018.extend(benign_2022)
    # benign_2018.extend(benign_2023)
    # 
    # 
    # # malware = [row[0] for row in train_dataset[6000:12000]]
    # malware_2018 = [row[0] for row in train_dataset[6800:7000]]
    # malware_2019 = [row[0] for row in train_dataset[7800:8000]]
    # malware_2020 = [row[0] for row in train_dataset[8800:9000]]
    # malware_2021 = [row[0] for row in train_dataset[9800:10000]]
    # malware_2022 = [row[0] for row in train_dataset[10800:11000]]
    # malware_2023 = [row[0] for row in train_dataset[11800:12000]]
    # malware_labels = [1] * 1200
    # malware_2018.extend(malware_2019)
    # malware_2018.extend(malware_2020)
    # malware_2018.extend(malware_2021)
    # malware_2018.extend(malware_2022)
    # malware_2018.extend(malware_2023)
    # 
    # #垂直拼接
    # benign_2018.extend(malware_2018)
    # X_dataset = np.array(benign_2018)
    # benign_labels.extend(malware_labels)
    # Y_dataset = np.array(benign_labels)
    # ftype = 'apigraph'
    # model_dir = '/data/b/guoqi/icse25/malwareGA/model/apigraph/'
    # model1 = load_test_model(f'{model_dir}{ftype}AdaBoost.pkl')
    # model2 = load_test_model(f'{model_dir}{ftype}RandomForest.pkl')
    #     # model2 = load_test_model(f'{model_dir}allSVM.pkl')
    # model3 = load_test_model(f'{model_dir}{ftype}knn_1.pkl')
    # model4 = load_test_model(f'{model_dir}{ftype}knn_3.pkl')
    # model5 = load_test_model(f'{model_dir}{ftype}MLP.h5')
    # # # #     #test
    # print("AdaBoost") # 0 1
    # cal_and_test_model(X_dataset, Y_dataset, model1)
    # print("RandomForest") # 0 1
    # cal_and_test_model(X_dataset, Y_dataset, model2)
    # #     print("knn_1") # 0 1
    # cal_and_test_model(X_dataset, Y_dataset, model3)
    # print("knn_3") # 0 1
    # cal_and_test_model(X_dataset, Y_dataset, model4)
    # print("MLP") #  float[0] 1
    # cal_and_test_model(X_dataset, Y_dataset, model5, '123')
    #     # print("RF substitute")
    #     knn_1 = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}knn_3.pkl')
    #     # MLP_knn_1 = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}MLP_SVM.h5', shape)
    #     # cal_and_test_substitute_model(X_dataset, knn_1, MLP_knn_1)
    #     #
    #     shape = 121
    #     MLP_knn_1 = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}MLP_knn_3.h5', shape)
    #     cal_and_test_substitute_model(X_dataset, knn_1, MLP_knn_1)

        #
        #
        # print("knn_3 substitute")
        # knn_3 = load_test_model(f'./430features_6yearsdataset_1000samples_new/{feature_type}knn_3.pkl')
        # MLP_knn_3 = load_test_model(f'./430features_6yearsdataset_1000samples_new/{feature_type}MLP_knn_3.h5', shape)
        # cal_and_test_substitute_model(X_dataset, knn_3, MLP_knn_3)
        #
        # print("RandomForest substitute")
        # rf = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}_usxRandomForest.pkl')
        # MLP_rf = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}_usxMLP_RandomForest.h5', shape)
        # cal_and_test_substitute_model(X_dataset, rf, MLP_rf)
        #
        # print("AdaBoost substitute")
        # ab = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}_usxAdaBoost.pkl')
        # MLP_ab = load_test_model(f'./430features_6yearsdataset_1000samples/{feature_type}_usxMLP_AdaBoost.h5', shape)
        # cal_and_test_substitute_model(X_dataset, ab, MLP_ab)







    #     ab_path = f'./430features_6yearsdataset_all/{feature_type}AdaBoost.pkl'
    # ab = joblib.load(ab_path)
    # train_substitute_Model(X_dataset, ab, 'MLP', 'AdaBoost', './430features_6yearsdataset_all/', feature_type)
    #
    # rf_path = f'./430features_6yearsdataset_all/{feature_type}RandomForest.pkl'
    # rf = joblib.load(rf_path)
    # train_substitute_Model(X_dataset, rf, 'MLP', 'RandomForest', './430features_6yearsdataset_all/', feature_type)
    #
    # knn3_path = f'./430features_6yearsdataset_all/{feature_type}knn_3.pkl'
    # knn3 = joblib.load(knn3_path)
    # train_substitute_Model(X_dataset, knn3, 'MLP', 'knn_3', './430features_6yearsdataset_all/', feature_type)

    # # model2 = dnn_train_model(X_dataset, Y_dataset, './430features_3yearsdataset_all/')


    # X_train = np.vstack((X_benign, X_malware))
    # Y_train = np.hstack((Y_benign, Y_malware))
    # print(X_train.shape)
    # print(Y_train.shape)
    # import_features = find_important_features_based_on_decisiontree(X_train, Y_train, 1000)
    # train_model(X_train, Y_train, 'SVM', model_output_path='/data/a/shiwensong/model/all_features/')
    # train_model(X_train, Y_train, 'RandomForest', model_output_path='/data/a/shiwensong/model/all_features/')


    # obtain features
    # X_benign_features = X_benign[1383]
    # X_malware_features = X_malware[48]
    # for i in range(len(X_benign_features)):
    #     print(X_benign_features[i], X_malware_features[i])
    #     if X_benign_features[i] != X_malware_features[i]:
    #         print("error")
    #     if X_benign_features[i] != 0.0:
    #         print("ooo")

    # print(mode_path_benign)
    # print(mode_path_malware)
    #choose important features
    # deal_with_benign_dataset()

    #test model
    # types = ['degree', 'katz', 'closeness', 'harmonic', 'avg', 'family', 'package']

    # X_test_benign_2019, Y_test_benign_2019 = test_dataset_benign()#benign
    # X_test_malware_2023, Y_test_malware_2023 = test_dataset_malware()  # malware
    # # # cnt_benign_error1 = 0 #FN
    # # # cnt_benign_error2 = 0  # FN
    # # # cnt_benign_error3 = 0  # FN
    # # # cnt_benign_error4 = 0  # FN
    # # # cnt_benign_right1 = 0 #TN
    # # # cnt_benign_right2 = 0  # TN
    # # # cnt_benign_right3 = 0  # TN
    # # # cnt_benign_right4 = 0  # TN
    # model_path1 = '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_all/RandomForest.pkl'
    # model_path2 = '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_all/MLP_RandomForest.h5'
    # model_path3 = '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_all/AdaBoost.pkl'
    # model_path4 = '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_all/MLP_AdaBoost.h5'
    # model_path5 = '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_all/AdaBoost.pkl'
    # model_path6 = '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_all/MLP_AdaBoost.h5'
    # model_path7 = '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_all/knn_3.pkl'
    # model_path8 = '/data/b/shiwensong/project/malwareGA/task/430features_6yearsdataset_all/MLP_knn_3.h5'
    # # # model1 = joblib.load(model_path1)
    # # # model2 = joblib.load(model_path2)
    # # # model3 = joblib.load(model_path3)
    # # # model4 = joblib.load(model_path4)
    # # #
    # model1 = load_test_model(model_path1)
    # model2 = load_test_model(model_path2)
    # model3 = load_test_model(model_path3)
    # model4 = load_test_model(model_path4)
    # model5 = load_test_model(model_path5)
    # model6 = load_test_model(model_path6)
    # model7 = load_test_model(model_path7)
    # model8 = load_test_model(model_path8)
    #
    # # # #拼接良性恶意
    # X_test = np.vstack((X_test_benign_2019, X_test_malware_2023))
    # Y_test = np.hstack((Y_test_benign_2019, Y_test_malware_2023))
    # res1 = model1.predict(X_test)
    # res2 = model2.predict(X_test)
    # #
    # cnt_error1 = 0
    # print("res1", len(res1))
    # for i in range(len(res1)):
    #     print("res1", res1[i], "res2", res2[i])
    #     if res1[i] == 1 and res2[i][0] < 0:
    #         cnt_error1 = cnt_error1 + 1
    #
    #     if res1[i] == 0 and res2[i][0] > 0:
    #         cnt_error1 = cnt_error1 + 1
    #
    # print("RF cnt_error", cnt_error1)
    #
    # res3 = model3.predict(X_test)
    # res4 = model4.predict(X_test)
    # #
    # cnt_error3 = 0
    # print("res3", len(res3))
    # for i in range(len(res3)):
    #     print("res3", res3[i], "res4", res4[i])
    #     if res3[i] == 1 and res4[i][0] < 0:
    #         cnt_error3 = cnt_error3 + 1
    #
    #     if res3[i] == 0 and res4[i][0] > 0:
    #         cnt_error3 = cnt_error3 + 1
    #
    # print("AB cnt_error", cnt_error3)
    #
    # res5 = model5.predict(X_test)
    # res6 = model6.predict(X_test)
    # #
    # cnt_error5 = 0
    # print("res5", len(res5))
    # for i in range(len(res5)):
    #     print("res5", res5[i], "res6", res6[i])
    #     if res5[i] == 1 and res6[i][0] < 0:
    #         cnt_error5 = cnt_error5 + 1
    #
    #     if res5[i] == 0 and res6[i][0] > 0:
    #         cnt_error5 = cnt_error5 + 1
    #
    # print("AB cnt_error", cnt_error5)
    #
    # res7 = model7.predict(X_test)
    # res8 = model8.predict(X_test)
    # #
    # cnt_error7 = 0
    # print("res7", len(res7))
    # for i in range(len(res7)):
    #     print("res7", res5[i], "res8", res8[i])
    #     if res7[i] == 1 and res8[i][0] < 0:
    #         cnt_error7 = cnt_error7 + 1
    #
    #     if res7[i] == 0 and res8[i][0] > 0:
    #         cnt_error7 = cnt_error7 + 1
    #
    # print("AB cnt_error", cnt_error7)


    # 
    # # 现在，MLP2 拥有了 MLP 的权重，但最后一层没有激活函数
    # 
    # Y_predict_benign1 = model1.predict(X_test_benign_2019)
    # Y_predict_benign2 = model2.predict(X_test_benign_2019)
    # Y_predict_benign3 = model3.predict(X_test_benign_2019)
    # Y_predict_benign4 = model4.predict(X_test_benign_2019)
    # print("predict1", Y_predict_benign1.shape)
    # print("predict2", Y_predict_benign2.shape)
    # print("predict3", Y_predict_benign3.shape)
    # print("predict4", Y_predict_benign4.shape)
    # for i in range(len(Y_predict_benign1)):
    #     Y_pred1 = Y_predict_benign1[i]
    #     Y_pred2 = Y_predict_benign2[i]
    #     # Y_pred3 = Y_predict_benign3[i]
    #     # Y_pred4 = Y_predict_benign4[i]
    #     if Y_pred1 == 1:
    #         cnt_benign_error1 = cnt_benign_error1 + 1
    #
    #     if Y_pred2 == 1:
    #         cnt_benign_error2 = cnt_benign_error2 + 1

        # if Y_pred3 == 1:
        #     cnt_benign_error3 = cnt_benign_error3 + 1
        #
        # if Y_pred4 == 1:
        #     cnt_benign_error4 = cnt_benign_error4 + 1

        # if Y_pred1 == 0:
        #     cnt_benign_right1 = cnt_benign_right1 + 1
        #
        # if Y_pred2 == 0:
        #     cnt_benign_right2 = cnt_benign_right2 + 1

        # if Y_pred3 == 0:
        #     cnt_benign_right3 = cnt_benign_right3 + 1
        #
        # if Y_pred4 == 0:
        #     cnt_benign_right4 = cnt_benign_right4 + 1



    # Y_predict_malware1 = model1.predict(X_test_malware_2023)
    # Y_predict_malware2 = model2.predict(X_test_malware_2023)
    # Y_predict_malware3 = model3.predict(X_test_malware_2023)
    # Y_predict_malware4 = model4.predict(X_test_malware_2023)
    # cnt_malware_error1 = 0 #FP
    # cnt_malware_error2 = 0  # FP
    # cnt_malware_error3 = 0  # FP
    # cnt_malware_error4 = 0  # FP
    # cnt_malware_right1 = 0 #TP
    # cnt_malware_right2 = 0  # TP
    # cnt_malware_right3 = 0  # TP
    # cnt_malware_right4 = 0  # TP
    #
    # print("predict1", Y_predict_malware1.shape)
    # print("predict2", Y_predict_malware2.shape)
    # print("predict3", Y_predict_malware3.shape)
    # print("predict4", Y_predict_malware4.shape)
    # for i in range(len(Y_predict_malware1)):
    #     Y_pred1 = Y_predict_malware1[i]
    #     Y_pred2 = Y_predict_malware2[i]
    #     Y_pred3 = Y_predict_malware3[i]
    #     Y_pred4 = Y_predict_malware4[i]
    #     if Y_pred1 == 0:
    #         cnt_malware_error1 = cnt_malware_error1 + 1
    #
    #     if Y_pred2 == 0:
    #         cnt_malware_error2 = cnt_malware_error2 + 1
    #
    #     if Y_pred3 == 0:
    #         cnt_malware_error3 = cnt_malware_error3 + 1
    #
    #     if Y_pred4 == 0:
    #         cnt_malware_error4 = cnt_malware_error4 + 1
    #
    #     if Y_pred1 == 1:
    #         cnt_malware_right1 = cnt_malware_right1 + 1
    #
    #     if Y_pred2 == 1:
    #         cnt_malware_right2 = cnt_malware_right2 + 1
    #
    #     if Y_pred3 == 1:
    #         cnt_malware_right3 = cnt_malware_right3 + 1
    #
    #     if Y_pred4 == 1:
    #         cnt_malware_right4 = cnt_malware_right4 + 1
    # #计算f1 score presicion recall accuracy
    # print("result 1")
    # print(cnt_malware_error1)
    # print(cnt_malware_right1)
    # print(cnt_benign_error1)
    # print(cnt_benign_right1)
    #
    # print("result 2")
    # print(cnt_malware_error2)
    # print(cnt_malware_right2)
    # print(cnt_benign_error2)
    # print(cnt_benign_right2)
    # 
    # print("result 3")
    # print(cnt_malware_error3)
    # print(cnt_malware_right3)
    # print(cnt_benign_error3)
    # print(cnt_benign_right3)
    # 
    # print("result 4")
    # print(cnt_malware_error4)
    # print(cnt_malware_right4)
    # print(cnt_benign_error4)
    # print(cnt_benign_right4)
    # 
    # 
    # print(len(X_test_benign_2019))
    # print(len(X_test_malware_2023))
    #
    # precision_benign_score = cnt_benign_right/ (cnt_benign_right + cnt_benign_error)
    # recall_score = cnt_benign_right / (cnt_benign_right + cnt_malware_error)
    # #精确率是指在被识别为正类的样本中，实际为正类的比例
    # precision_score = cnt_benign_right / (cnt_benign_right + cnt_benign_error)
    # print("f1 score:", 2 * cnt_benign_right / (2 * cnt_benign_right + cnt_malware_error + cnt_benign_error))
    # print("precision:", precision_score)
    # print("recall:", recall_score)
    # print("accuracy:", cnt_malware_right + cnt_benign_right / (cnt_malware_right + cnt_benign_right + cnt_malware_error + cnt_benign_error))

    # deal_with_benign_dataset()
    
    # #degree
    # X_train_degree_benign, Y_train_degree_benign, X_test_degree_benign, Y_test_degree_benign = obtain_manyyears_features(0, 'degree')
    # X_train_degree_malware, Y_train_degree_malware, X_test_degree_malware, Y_test_degree_malware = obtain_manyyears_features(1, 'degree')
    # X_train_degree = np.vstack((X_train_degree_benign, X_train_degree_malware))
    # Y_train_degree = np.hstack((Y_train_degree_benign, Y_train_degree_malware))
    # degree_import_features = find_important_features_based_on_decisiontree(X_train_degree, Y_train_degree, 300)
    #
    # #katz
    # X_train_katz_benign, Y_train_katz_benign, X_test_katz_benign, Y_test_katz_benign = obtain_manyyears_features(0, 'katz')
    # X_train_katz_malware, Y_train_katz_malware, X_test_katz_malware, Y_test_katz_malware = obtain_manyyears_features(1, 'katz')
    # X_train_katz = np.vstack((X_train_katz_benign, X_train_katz_malware))
    # Y_train_katz = np.hstack((Y_train_katz_benign, Y_train_katz_malware))
    # katz_import_features = find_important_features_based_on_decisiontree(X_train_katz, Y_train_katz, 300)
    #
    # #harmonic
    # X_train_harmonic_benign, Y_train_harmonic_benign, X_test_harmonic_benign, Y_test_harmonic_benign = obtain_manyyears_features(0, 'harmonic')
    # X_train_harmonic_malware, Y_train_harmonic_malware, X_test_harmonic_malware, Y_test_harmonic_malware = obtain_manyyears_features(1, 'harmonic')
    # X_train_harmonic = np.vstack((X_train_harmonic_benign, X_train_harmonic_malware))
    # Y_train_harmonic = np.hstack((Y_train_harmonic_benign, Y_train_harmonic_malware))
    # harmonic_import_features = find_important_features_based_on_decisiontree(X_train_harmonic, Y_train_harmonic, 300)
    #
    # #closeness
    # X_train_closeness_benign, Y_train_closeness_benign, X_test_closeness_benign, Y_test_closeness_benign = obtain_manyyears_features(0, 'closeness')
    # X_train_closeness_malware, Y_train_closeness_malware, X_test_closeness_malware, Y_test_closeness_malware = obtain_manyyears_features(1, 'closeness')
    # X_train_closeness = np.vstack((X_train_closeness_benign, X_train_closeness_malware))
    # Y_train_closeness = np.hstack((Y_train_closeness_benign, Y_train_closeness_malware))
    # closeness_import_features = find_important_features_based_on_decisiontree(X_train_closeness, Y_train_closeness, 300)
    #
    # #取交集
    # intersect12 = np.intersect1d(degree_import_features, katz_import_features)
    # intersect123 = np.intersect1d(intersect12, harmonic_import_features)
    # intersect1234 = np.intersect1d(intersect123, closeness_import_features)
    # print(intersect12.shape)
    # print(intersect123.shape)
    # print(intersect1234.shape)
    #
    # #取并集
    # union12 = np.union1d(degree_import_features, katz_import_features)
    # union34 = np.union1d(harmonic_import_features, closeness_import_features)
    # union1234 = np.union1d(union12, union34)
    # print(union12.shape)
    # print(union34.shape)
    # print(union1234.shape)
    #
    # #保存到文件
    # sensitive_apis = obtain_sensitive_apis()
    # with open('new_important_sensitive_apis_based_on_dt.txt', 'w') as file:
    #     for i in union1234:
    #         line = sensitive_apis[i]
    #         file.write(line + '\n')  # 在每个字符串后添加换行符
            
    
    
