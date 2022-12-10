from collections import defaultdict
import numpy as np
import paddle
import paddle.nn.functional as F
import scipy.io
from sklearn.preprocessing import label_binarize
from os import path
import os

from load_data import load_twitch, load_fb100, load_twitch_gamer, DATAPATH
from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url

from homophily import our_measure, edge_homophily_edge_idx


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives paddle tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(dataname, sub_dataname=''):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname == "wiki":
        dataset = load_wiki()
    else:
        raise ValueError('Invalid dataname')
    return dataset

def load_wiki():

    dataset = NCDataset("wiki") 
    features = np.load(f'{DATAPATH}wiki_features2M.npy')
    edges = np.load(f'{DATAPATH}wiki_edges2M.npy')#.T
    row, col = edges
    print(f"edges shape: {edges.shape}")
    label = np.load(f'{DATAPATH}wiki_views2M.npy') 
    num_nodes = label.shape[0]

    print(f"features shape: {features.shape[0]}")
    print(f"Label shape: {label.shape[0]}")
    dataset.graph = {"edge_index": edges, 
                     # "edge_feat": None, 
                     "node_feat": features, 
                     "num_nodes": num_nodes}
    dataset.label = label 
    return dataset 
