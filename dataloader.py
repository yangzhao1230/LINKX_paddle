import os
import os.path as osp
import numpy as np


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, seed=520):
    """ randomly splits label into train/valid/test splits """

    labeled_nodes = np.where(label != -1)[0]

    n = labeled_nodes.shape[0]  
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    np.random.seed(seed)
    perm = np.random.permutation(n)

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx



class WikiDataset(object):
    def __init__(self, path):
        self.path = path       # Data path
        self.graph = None      # pgl graph object
        self.num_nodes = None  # Number of graphs
        self._load_data()      # Load Data

    def _load_data(self):
        x_path = osp.join(self.path, "wiki_features2M.npy")
        y_path = osp.join(self.path, "wiki_views2M.npy")
        edges_path = osp.join(self.path, "wiki_edges2M.npy")

        # load data
        features = np.load(x_path)
        edge_index = np.load(edges_path)
        self.num_nodes = features.shape[0]
        self.y = np.load(y_path)
        print("=== Sucessfully Load Data! ===")

        # build graph
        # print(edge_index.shape)
        # print(edge_index.T.shape)
        self.graph = Graph(num_nodes=self.num_nodes,edges=edge_index,node_feat={"feat": features})

    def generate_split(self):
        self.train_idx, self.val_idx, self.test_idx = \
            rand_train_test_idx(label=self.y)
            
if __name__ == '__main__':
    dataset = WikiDataset(path="./data")
    dataset.generate_split()

    print("num of nodes:",dataset.graph.num_nodes)
    print("num of edges:",dataset.graph.edges.shape[1])
    print("dimension of feature:",dataset.graph.node_feat["feat"].shape)
    print("class of labels:", int(max(dataset.y))+1)
    print("train Examples:",len(dataset.train_idx))
    print("val Examples:",len(dataset.val_idx))
    print("test Examples:",len(dataset.test_idx))