# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import dgl
def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad 
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, idx, attn_bias, attn_edge_type, rel_pos, in_degree, out_degree,
                  x,task_num,task, edge_input, y,bg,smiles):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.rel_pos = attn_bias, attn_edge_type, rel_pos
        self.edge_input = edge_input
        self.bg=bg
        self.task_num=task_num
        self.task=task
        self.smiles=smiles

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = self.in_degree.to(
            device), self.out_degree.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.attn_edge_type, self.rel_pos = self.attn_bias.to(
            device), self.attn_edge_type.to(device), self.rel_pos.to(device)
        self.edge_input = self.edge_input.to(device)
        self.task_num=self.task_num.to(device)
        self.task=self.task.to(device)
        self.bg=self.bg.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)


def collator(items, max_node=512, multi_hop_max_dist=20, rel_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.rel_pos, item.in_degree,
              item.out_degree, item.x, item.task_num, item.task,
                item.bg,item.edge_input[:, :, :multi_hop_max_dist, :], item.y,item.smiles)
                  for item in items]
    idxs, attn_biases, attn_edge_types, rel_poses, in_degrees, out_degrees, xs,task_num,task,bg, edge_inputs, ys, smiles= zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][rel_poses[idx] >= rel_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    # print(len(ys))
    y = torch.cat(ys)
    # print(y.shape)
    # smiles=torch.cat(smiles)
    # print(len(smiles),type(smiles))
    bg=dgl.batch(bg)
    _=bg.ndata.pop('atom')
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    task_num=torch.cat(task_num)
    task=torch.cat(task)
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num)
                        for i in rel_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                           for i in out_degrees])
    return Batch(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        rel_pos=rel_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        task_num=task_num,
        task=task,
        edge_input=edge_input,
        y=y,
        bg=bg,
        smiles=smiles,
    )
