import datetime
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score
import torch
import torch.nn.functional as F
import numpy as np
import random
from dgl.readout import sum_nodes
from torch import nn
import os


class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size

    return_weight: bool
        Defalt: False
    """

    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super(WeightAndSum, self).__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight = return_weight
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])
        self.shared_weighting = self.atom_weight(self.in_feats)

    def forward(self, bg, feats):
        """Compute molecule representations out of atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            B Batched DGLGraphs for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules
        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        atom_weight for each atom
        """
        feat_list = []
        atom_list = []
        # _=bg.ndata.pop('atom')
        # _=bg.ndata.pop('etype')
        # cal specific feats
        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata['w'] = weight
                specific_feats_sum = sum_nodes(bg, 'h', 'w')
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)

        # cal shared feats
        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')
        # feat_list.append(shared_feats_sum)
        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            else:
                return feat_list
        else:
            return shared_feats_sum

    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

import pickle
class Multi_Task(nn.Module):
    def __init__(self,task_num,feat_dim, return_mol_embedding=False, return_weight=True,
                 classifier_hidden_feats=128, dropout=0.):
        super(Multi_Task, self).__init__()
        self.task_num = task_num
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(feat_dim, self.task_num, return_weight=self.return_weight)
        self.fc_in_feats = feat_dim
        self.return_mol_embedding = return_mol_embedding

        self.fc_layers1 = nn.ModuleList(
            [self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])

        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])

    def forward(self,batched_data, feats, norm=None):
        bg=batched_data.bg   
        data_x = batched_data.x

        padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1
        feats=feats[:, 1:, :]
        #get node feats
        unfold_padding_mask=padding_mask.reshape(-1)
        unfold_feats=feats.reshape(-1,self.fc_in_feats)
        node_idx=torch.nonzero(unfold_padding_mask==0).squeeze()
        node_feat_list=unfold_feats[node_idx,:]
        #print(feats.shape,unfold_feats.shape,node_feat_list.shape)
        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feat_list)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feat_list)

        for i in range(self.task_num):
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)

            predict = self.output_layer1[i](h3)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)
        pred=prediction_all[:,-3]
        # print(a)
        # generate toxicity fingerprints
        if self.return_mol_embedding:
            return feats_list[0]
        else:
            # generate atom weight
            if self.return_weight:
                return prediction_all, atom_weight_list
            # just generate prediction
            return prediction_all

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats)
        )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
            nn.Linear(hidden_feats, out_feats)
        )

