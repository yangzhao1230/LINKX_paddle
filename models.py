# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import scipy.sparse
from tqdm import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils.sparse_nn import Sparse_Linear

class LINKX(nn.Layer):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()	
        # self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpA = Sparse_MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    # TODO
    # def reset_parameters(self):	
    #     self.mlpA.reset_parameters()	
    #     self.mlpX.reset_parameters()
    #     self.W.reset_parameters()
    #     self.mlp_final.reset_parameters()	

    def forward(self, data):	
        m = data.graph['num_nodes']	
        feat_dim = data.graph['node_feat']	
        row, col = data.graph['edge_index']	
        row = row-row.min()
        # A = SparseTensor(row=row, col=col,	
        #          sparse_sizes=(m, self.num_nodes)
        #                 ).to_torch_sparse_coo_tensor()
        indices = [row, col]
        values = paddle.ones([row.shape[0]])
        dense_shape = [m, self.num_nodes]
        A = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, dtype='float32')
        # A = A.to_dense()
        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(data.graph['node_feat'], input_tensor=True)
        x = paddle.concat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        return x

class MLP(nn.Layer):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.LayerList()
        self.bns = nn.LayerList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1D(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1D(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    # TODO 
    # def reset_parameters(self):
    #     for lin in self.lins:
    #         lin.reset_parameters()
    #     for bn in self.bns:
    #         bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class Sparse_MLP(nn.Layer):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(Sparse_MLP, self).__init__()
        self.lins = nn.LayerList()
        self.bns = nn.LayerList()
        # FIXME only support num_layer == 1
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(Sparse_Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1D(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1D(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    # data is a sparse matrix
    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
