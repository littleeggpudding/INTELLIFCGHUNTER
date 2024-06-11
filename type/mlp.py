#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the Multi-Layer Perceptron (MLP).
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from logging import Logger
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np


# class MLP(torch.nn.Module):
#     def __init__(self,
#                  in_channels:int,
#                  hidden_channels:int,
#                  out_channels:int,
#                  attention:bool=False,
#                  dropout_ratio:float=0.2):
#         super().__init__()
#
#         # Cannot be used if in_channels is huge (e.g., 125787 in Drebin). Otherwise, it causes GPU out of memory.
#         self.attention = attention
#         if self.attention:
#             if in_channels > 2000:
#                 raise ValueError('Cannot use attention in MLP if in_channels is huge.')
#             self.att_mlp_layer = nn.Linear(in_channels, in_channels)
#
#         self.pred_linear = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(dropout_ratio),
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(dropout_ratio),
#             nn.Linear(hidden_channels, out_channels)
#         )
#
#         self.loss_fn = torch.nn.CrossEntropyLoss()
#
#     def forward(self, x, y):
#         if self.attention:
#             QK = self.att_mlp_layer(x)
#             QK = F.softmax(QK, dim=1)
#             x = torch.mul(x, QK)
#         y_score = self.pred_linear(x)
#         loss = self.loss_fn(y_score, y)
#         return loss
#
#     def predict(self, x):
#         if self.attention:
#             QK = self.att_mlp_layer(x)
#             QK = F.softmax(QK, dim=1)
#             x = torch.mul(x, QK)
#         y_score = self.pred_linear(x)
#         y_pred = torch.argmax(y_score, dim=1)
#         return y_pred
#
#     # def predict(self, x):
#     #     if isinstance(x, np.ndarray):
#     #         x = torch.tensor(x, dtype=torch.float32)
#     #     x = x.to('cuda')
#     #     with torch.no_grad():
#     #         y_score = self.forward(x)
#     #         y_pred = torch.argmax(y_score, dim=1)
#     #     return y_pred.cpu().numpy()  # Ensure conversion to NumPy does not fail
#
#     # def predict_proba(self, x):
#     #     probabilities = self.forward(x)
#     #     _, y_pred = torch.max(probabilities, 1)
#     #     return y_pred
#
#     def score(self, x):
#         if self.attention:
#             QK = self.att_mlp_layer(x)
#             QK = F.softmax(QK, dim=1)
#             x = torch.mul(x, QK)
#         y_score = self.pred_linear(x)
#         return y_score

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 attention: bool = False,
                 dropout_ratio: float = 0.2):
        super().__init__()

        self.attention = attention
        if self.attention:
            if in_channels > 2000:
                raise ValueError('Cannot use attention in MLP if in_channels is huge.')
            self.att_mlp_layer = nn.Linear(in_channels, in_channels)

        self.pred_linear = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_channels, out_channels)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        if self.attention:
            QK = self.att_mlp_layer(x)
            QK = F.softmax(QK, dim=1)
            x = torch.mul(x, QK)
        return self.pred_linear(x)

    def predict(self, x):
        # Convert input from NumPy array to Tensor if necessary
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to('cuda')

        # Forward pass without computing gradients
        with torch.no_grad():
            y_score = self.forward(x)
            y_pred = torch.argmax(y_score, dim=1)

        # Convert prediction back to NumPy array for compatibility with SHAP
        return y_pred.cpu().numpy()

    def score(self, x):
        # This function can be adjusted similarly to 'predict' if needed for SHAP
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to('cuda')
        if self.attention:
            QK = self.att_mlp_layer(x)
            QK = F.softmax(QK, dim=1)
            x = torch.mul(x, QK)
        y_score = self.pred_linear(x)
        return y_score

def mlp_train(model:MLP, logger:Logger, train_data, val_data=None, test_data=None, model_path:str=None, evaluation=True):
    # device to run the model
    device_id = 0
    lr = 1e-3
    epochs = 100
    batch_size = 100
    device = get_device(device_id)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    ret_val = defaultdict(lambda: "Not present")
    ret_val['f1'] = 0.0
    ret_test = defaultdict(lambda: "Not present")

    # Early stopping
    patitence = 12
    tolerance = 0.001
    no_improvement_count = 0
    # Total time
    total_train_time = 0.
    total_val_time = 0.

    for epoch in range(epochs):
        model.train()
        train_start_time = time.time()
        all_loss = 0.0
        for batch in train_loader:
            x = batch[0].float().to(device)
            y = batch[1].long().to(device)

            loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss += loss
        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        logger.info('Epoch: {}, Loss: {:.4f}, Time: {:.2f}s'.format(epoch, all_loss, train_end_time-train_start_time))
        total_train_time += train_time

        if evaluation:
            # Validation
            val_start_time = time.time()
            ret_val_tmp = mlp_evaluate(val_data, model, device_id)
            val_end_time = time.time()
            val_time = val_end_time - val_start_time
            total_val_time += val_time
            logger.debug("Validation:[f1, recall, precision, accuracy, time]=[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(ret_val_tmp['f1'], ret_val_tmp['accuracy'], ret_val_tmp['precision'], ret_val_tmp['f1'], val_time))

            logger.warning("Total training time: {:.2f}s, Total validation time: {:.2f}s".format(total_train_time, total_val_time))

            if ret_val_tmp['f1'] > ret_val['f1'] + tolerance:
                ret_val = ret_val_tmp
                test_start_time = time.time()
                ret_test = mlp_evaluate(test_data, model, device_id)
                test_end_time = time.time()
                torch.save(model, model_path)
                # Reset the no_improvement count
                no_improvement_count = 0

                logger.debug("Testing:[f1, recall, precision, accuracy, time]=[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2}s]".format(ret_test['f1'], ret_test['recall'], ret_test['precision'], ret_test['accuracy'], test_end_time-test_start_time))
            else:
                no_improvement_count += 1
                # Early stopping
                if no_improvement_count >= patitence:
                    logger.warning("Early stopping at epoch {}".format(epoch))
                    break

    return ret_val, ret_test

def mlp_evaluate(data, model:MLP, device_id:int):
    # device to run the model
    device = get_device(device_id)

    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=4)

    model.eval()
    y_pred, y_true = [], []
    for batch in loader:
        x = batch[0].to(device).float()
        y = batch[1].to(device).float()

        with torch.no_grad():
            y_pred_x = model.predict(x)     

        y_pred.append(y_pred_x)
        y_true.append(y)

    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    y_true = torch.cat(y_true, dim=0).cpu().numpy()

    ret = eval_metrics(y_true, y_pred)

    return ret

def get_device(dev=None):
    """ get device """
    if dev == -1:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        if dev is None:
            # default: use GPU 0
            dev = 0
        device = torch.device(dev)
    else:
        device = torch.device('cpu')

    return device

def eval_metrics(y_true, y_pred):
    ret = defaultdict(lambda: "Not present")
    ret['auc'] = roc_auc_score(y_true=y_true, y_score=y_pred)

    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_true[i]:
                TP += 1
            else:
                TN += 1
        else:
            if y_true[i]:
                FN += 1
            else:
                FP += 1

    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    ret['TP'], ret['TN'], ret['FP'], ret['FN'] = TP, TN, FP, FN
    ret['precision'], ret['recall'], ret['f1'], ret['accuracy'] = metric2scores(TP, FP, TN, FN)
    ret['tpr'], ret['fpr'] = TPR, FPR

    return ret

def metric2scores(TP, FP, TN, FN):
    correct = TP + TN
    total = correct + FP + FN
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    accuracy = correct / total

    f1 = calculate_f1(precision, recall, 1)

    return precision, recall, f1, accuracy

def calculate_f1(precision, recall, beta):
    """ calculate f1 score """
    try:
        return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    except ZeroDivisionError:
        return 0.
