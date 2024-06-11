#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
This file defines the GNN network for MsDroid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class MsDroidNet(nn.Module):
    """ Customized GNN Network for MsDroid """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MsDroidNet, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # hard code the number of layers to 3 following the paper
        self.num_layers = 3

        self.convs.append(self.gnn_layer(input_dim, hidden_dim))
        for _ in range(2):
            self.convs.append(self.gnn_layer(hidden_dim, hidden_dim))
            self.batch_norms.append(pyg_nn.BatchNorm(hidden_dim))

        self.post_mp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), \
                                     nn.Dropout(0.25), nn.Linear(hidden_dim, output_dim))

    def gnn_layer(self, input_dim: int, hidden_dim: int):
        """ GNN layer """
        return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), \
                                            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, x, edge_index, batch):
        """ Forward pass """
        node_x = self.get_node_reps(x, edge_index, batch)
        graph_x = torch.cat((pyg_nn.global_mean_pool(node_x, batch), pyg_nn.global_max_pool(node_x, batch)), dim=1)
        pred = self.get_pred(graph_x)

        return pred

    def get_node_reps(self, x, edge_index, batch):
        """ Get node representations """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.25, training=self.training)
            if i != self.num_layers - 1:
                x = self.batch_norms[i](x)

        return x

    def get_graph_rep(self, x, edge_index, batch):
        """ Get graph representation """
        node_x = self.get_node_reps(x, edge_index, batch)
        graph_x = torch.cat((pyg_nn.global_mean_pool(node_x, batch), pyg_nn.global_max_pool(node_x, batch)), dim=1)

        return graph_x

    def get_pred(self, graph_x):
        """ Get prediction """
        pred = self.post_mp(graph_x)
        readout = F.log_softmax(pred, dim=1)

        return readout  