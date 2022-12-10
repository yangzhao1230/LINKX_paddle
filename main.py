import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.optimizer import Adam, AdamW

from logger import Logger
from dataset import load_nc_dataset, NCDataset
from data_utils import normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, to_sparse_tensor
from parse import parse_method, parser_add_main_args
from batch_utils import nc_dataset_to_geo, geo_to_nc_dataset, AdjRowLoader, make_loader
import time

from dataloader import WikiDataset

np.random.seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
parser.add_argument('--train_batch', type=str, default='cluster', help='type of mini batch loading scheme for training GNN')
parser.add_argument('--no_mini_batch_test', action='store_true', help='whether to test on mini batches as well')
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--num_parts', type=int, default=100, help='number of partitions for partition batching')
parser.add_argument('--cluster_batch_size', type=int, default=1, help='number of clusters to use per cluster-gcn step')
parser.add_argument('--saint_num_steps', type=int, default=5, help='number of steps for graphsaint')
parser.add_argument('--test_num_parts', type=int, default=10, help='number of partitions for testing')
args = parser.parse_args()
print(args)

dataset = load_nc_dataset(args.dataset, args.sub_dataset)

dataset.label = np.expand_dims(dataset.label, -1)
# dataset.label = paddle.to_tensor(dataset.label)

split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
train_idx = split_idx['train']
# train_idx = paddle.to_tensor(train_idx)

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]
# print(n,c,d)

# TODO to undirected graph
# dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

train_loader, subgraph_loader = None, None

print(f"num nodes {n} | num classes {c} | num node feats {d}")

model = parse_method(args, dataset, n, c, d)
criterion = nn.NLLLoss()
eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

def train():
    model.train()

    total_loss = 0
    for tg_batch in train_loader:
        batch_train_idx = tg_batch.mask
        batch_dataset = geo_to_nc_dataset(tg_batch)
        # conver graph into tensor 
        batch_dataset.label = paddle.to_tensor(batch_dataset.label)
        batch_dataset.graph['node_feat'] = paddle.to_tensor(batch_dataset.graph['node_feat'])
        batch_dataset.graph['edge_index'] = paddle.to_tensor(batch_dataset.graph['edge_index'])
        # batch_dataset.graph['edge_feat'] = paddle.to_tensor(batch_dataset.graph['edge_feat'])
        batch_dataset.graph['num_nodes'] = paddle.to_tensor(batch_dataset.graph['num_nodes'])
        optimizer.clear_grad()
        # start = time.time()
        out = model(batch_dataset)
        # end = time.time()
        # print('Model Running time: %s Seconds'%(end-start))
        if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins'):
            if dataset.label.shape[1] == 1:
                # change -1 instances to 0 for one-hot transform
                # dataset.label[dataset.label==-1] = 0
                true_label = F.one_hot(batch_dataset.label, batch_dataset.label.max() + 1).squeeze(1)
            else:
                true_label = batch_dataset.label

            loss = criterion(out[batch_train_idx], true_label[batch_train_idx].to(out.dtype))
        else:
            out = F.log_softmax(out, axis=1)
            loss = criterion(
                out[batch_train_idx], batch_dataset.label.squeeze(1)[batch_train_idx])
        total_loss += loss
        loss.backward()
        optimizer.step()
 
    return total_loss

@paddle.no_grad()
def test():
    # needs a loader that includes every node in the graph
    model.eval()
    
    full_out = paddle.zeros([n, c])
    # with paddle.no_grad():
    for tg_batch in test_loader:
            node_ids = tg_batch.node_ids
            batch_dataset = geo_to_nc_dataset(tg_batch)
            # conver graph into tensor 
            batch_dataset.label = paddle.to_tensor(batch_dataset.label)
            batch_dataset.graph['node_feat'] = paddle.to_tensor(batch_dataset.graph['node_feat'])
            batch_dataset.graph['edge_index'] = paddle.to_tensor(batch_dataset.graph['edge_index'])
            # batch_dataset.graph['edge_feat'] = paddle.to_tensor(batch_dataset.graph['edge_feat'])
            batch_dataset.graph['num_nodes'] = paddle.to_tensor(batch_dataset.graph['num_nodes'])
            out = model(batch_dataset)
            full_out[node_ids] = out
    result = evaluate(model, dataset, split_idx, eval_func, result=full_out, sampling=args.sampling, subgraph_loader=subgraph_loader)
    logger.add_result(run, result[:])
    return result
    
### Training loop ###
for run in range(args.runs):
    train_idx = split_idx['train']
    # train_idx = paddle.to_tensor(train_idx)

    #print('making train loader')
    # start = time.time()
    train_loader = make_loader(args, dataset, train_idx)
    # end = time.time()
    # print('TrainLoader Running time: %s Seconds'%(end-start))
    # print(f"Len of TrainLoader:{len(train_loader)}")
    if not args.no_mini_batch_test:
        test_loader = make_loader(args, dataset, train_idx, test=True)
    else:
        test_idx = dataset.test_idx
        # test_idx = paddle.to_tensor(dataset.test_idx)
        test_loader = make_loader(args, dataset, test_idx, mini_batch = False)

    # TODO
    # model.reset_parameters()

    if args.adam:
        optimizer = Adam(model.parameters(), learning_rate=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(learning_rate=args.lr, parameters=model.parameters(), weight_decay=args.weight_decay)

    best_val = float('-inf')
    for epoch in range(args.epochs):
        # start = time.time()
        total_loss = train()
        # end = time.time()
        # print('Train Running time: %s Seconds'%(end-start))
        # start = time.time()
        if epoch % args.display_step == 0:
            result = test()
            if result[1] > best_val:
                # best_out = F.log_softmax(result[-1], axis=1)
                best_val = result[1]
        # end = time.time()
        # print('Test Running time: %s Seconds'%(end-start))
        

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch}, '
                  f'Loss: {total_loss.item()}, '
                  f'Train: {100 * result[0]}%, '
                  f'Valid: {100 * result[1]}%, '
                  f'Test: {100 * result[2]}%')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(return_counts=True)[1].float()/pred.shape[0])
    logger.print_statistics(run)

    # TODO
    # split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)

### Save results ###
best_val, best_test = logger.print_statistics()
filename = f'results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.method}," + f"{sub_dataset}" +
                    f"{best_val.mean():.3f} ± {best_val.std():.3f}," +
                    f"{best_test.mean():.3f} ± {best_test.std():.3f}\n")