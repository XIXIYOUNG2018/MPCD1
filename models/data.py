# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collator import collator
from wrapper import MyCHEMBLDataset

from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ogb
import ogb.lsc
import ogb.graphproppred
import numpy as np
from functools import partial
from utils.evalator import CHEMBLEvaluator,CLFEvaluator,REGEvaluator
import torch.nn as nn
dataset = None


def get_dataset(dataset_name='abaaba'):
    global dataset
    if dataset is not None:
        return dataset

    if dataset_name=='clf':
        dataset = {
            'num_class': 128,
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'auc',
            'metric_mode': 'max',
            'evaluator': CLFEvaluator(),  # same objective function, so reuse it
            'train_dataset': MyCHEMBLDataset( root='../../dataset/clf', split='train'),
            'valid_dataset': MyCHEMBLDataset( root='../../dataset/clf', split='val'),
            'test_dataset': MyCHEMBLDataset( root='../../dataset/clf', split='test'),
            'max_node': 128,
        }
    elif dataset_name=='reg':
        dataset = {
            'num_class': 128,
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'mae',
            'metric_mode': 'max',
            'evaluator': CLFEvaluator(),  # same objective function, so reuse it
            'train_dataset': MyCHEMBLDataset(root='../../dataset/reg', split='train'),
            'valid_dataset': MyCHEMBLDataset( root='../../dataset/reg', split='val'),
            'test_dataset': MyCHEMBLDataset( root='../../dataset/reg', split='test'),
            'max_node': 128,
        }
    else:
        root_path='../../dataset/'+dataset_name
        dataset = {
            'num_class': 128,
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': REGEvaluator(),  # same objective function, so reuse it
            'train_dataset': MyCHEMBLDataset( root=root_path, split='train'),
            'valid_dataset': MyCHEMBLDataset( root=root_path, split='val'),
            'test_dataset': MyCHEMBLDataset( root=root_path, split='test'),
            'max_node': 128,
        }

    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset


class GraphDataModule(LightningDataModule):

    def __init__(
        self,
        dataset_name: str = 'ogbg-molpcba',
        num_workers: int = 0,
        batch_size: int = 256,
        seed: int = 42,
        multi_hop_max_dist: int = 5,
        rel_pos_max: int = 1024,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.multi_hop_max_dist = multi_hop_max_dist
        self.rel_pos_max = rel_pos_max

    def setup(self, stage: str = None):

        self.dataset_train = self.dataset['train_dataset']
        self.dataset_val = self.dataset['valid_dataset']
        self.dataset_test = self.dataset['test_dataset']

    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
        )
        print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
        )
        print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, rel_pos_max=self.rel_pos_max),
        )
        print('len(test_dataloader)', len(loader))
        return loader
