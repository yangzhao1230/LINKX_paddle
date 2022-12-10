import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# import pgl
# from pgl import graph
from fake_pyg import Data
# from paddle_geometric.utils import to_undirected, sort_edge_index
# from paddle_geometric.data import NeighborSampler, ClusterData, ClusterLoader, Data, GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler, RandomNodeSampler
# from paddle_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset, NCDataset
from data_utils import normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, to_sparse_tensor
from parse import parse_method, parser_add_main_args


def nc_dataset_to_geo(dataset, idx):
    tg_data = Data()
    tg_data.x = dataset.graph['node_feat']
    tg_data.edge_index = dataset.graph['edge_index']
    # tg_data.edge_attr = dataset.graph['edge_feat']
    tg_data.y = dataset.label
    mask = np.zeros(tg_data.num_nodes, dtype=bool)
    mask[idx] = True
    tg_data.node_ids = np.arange(tg_data.num_nodes)
    tg_data.mask = mask
    return tg_data

def geo_to_nc_dataset(tg_data, name=''):
    dataset = NCDataset(name)
    dataset.label = tg_data.y
    dataset.graph['node_feat'] = tg_data.x
    dataset.graph['edge_index'] = tg_data.edge_index
    # dataset.graph['edge_feat'] = tg_data.edge_attr
    dataset.graph['num_nodes'] = dataset.graph['node_feat'].shape[0]
    return dataset

def sort_edge_index(edge_index):
    num_nodes = edge_index.shape[-1]
    idx = edge_index[0] * num_nodes + edge_index[1]
    perm = idx.argsort()
    return edge_index[:, perm]

class AdjRowLoader():
    def __init__(self, dataset, idx, num_parts=100, full_epoch=False):
        """
        if not full_epoch, then just return one chunk of nodes
        """
        self.dataset = dataset
        self.full_epoch = full_epoch
        n = dataset.graph['num_nodes']
        self.node_feat = dataset.graph['node_feat']
        self.edge_index = dataset.graph['edge_index']
        self.edge_index = sort_edge_index(self.edge_index)
        self.part_spots = [0]
        self.part_nodes = [0]
        self.idx = idx
        self.mask = np.zeros(dataset.graph['num_nodes'], dtype=bool)#, device=device)
        self.mask[idx] = True
        num_edges = self.edge_index.shape[1]
        approx_size = num_edges // num_parts
        approx_part_spots = list(range(approx_size, num_edges, approx_size))[:-1]
        for idx in approx_part_spots:
            curr_node = self.edge_index[0,idx].item()
            curr_idx = idx
            while curr_idx < self.edge_index.shape[1] and self.edge_index[0,curr_idx] == curr_node:
                curr_idx += 1
            self.part_nodes.append(self.edge_index[0, curr_idx].item())
            self.part_spots.append(curr_idx)
        self.part_nodes.append(n)
        self.part_spots.append(self.edge_index.shape[1])
    
    def __iter__(self):
        self.k = 0
        return self
    
    def __next__(self):
        if self.k >= len(self.part_spots)-1:
            raise StopIteration
            
        if not self.full_epoch:
            self.k = np.random.randint(len(self.part_spots)-1)
            
        # batch_edge_index = self.edge_index[:, self.part_spots[self.k]:self.part_spots[self.k+1]]
        # tg_data = graph.Graph(edges=batch_edge_index)

        tg_data = Data()
        batch_edge_index = self.edge_index[:, self.part_spots[self.k]:self.part_spots[self.k+1]]
        node_ids = list(range(self.part_nodes[self.k], self.part_nodes[self.k+1]))
        tg_data.node_ids = node_ids
        tg_data.edge_index = batch_edge_index
        batch_node_feat = self.node_feat[node_ids]
        tg_data.x = batch_node_feat
        # tg_data.edge_attr = None
        tg_data.y = self.dataset.label[node_ids]
        tg_data.num_nodes = len(node_ids)
        mask = self.mask[node_ids]
        tg_data.mask = mask
        self.k += 1
        
        if not self.full_epoch:
            self.k = float('inf')
        return tg_data
    

def make_loader(args, dataset, idx, mini_batch=True, test=False):
    if args.train_batch == 'row':
        loader = AdjRowLoader(dataset, idx, num_parts=args.num_parts, full_epoch=test)
        
    else:
        raise ValueError('Invalid train batching')
        
    return loader