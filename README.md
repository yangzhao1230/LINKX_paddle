# LINKX PaddlePaddle Implementation

This repository contains the PaddlePaddle implementation of LINKX (Large Scale Learning on Non-Homophilous Graphs, NeurIPS 2022).

## Environment Requirements

To run the code, you need to install the following:

- `paddle` - PaddlePaddle deep learning framework
- `pgl` - Paddle Graph Learning, a graph deep learning framework developed by Baidu

## Datasets

The model has been tested on the following dataset:

- Wiki - For more details regarding the dataset, visit the [Non-Homophily Large-Scale](https://github.com/CUAI/Non-Homophily-Large-Scale) GitHub repository.

## Instructions

To train the model, run the `main.py` script. You can modify the following parameters as needed:

```bash
python main.py --train_batch cluster \
               --no_mini_batch_test \
               --batch_size 10000 \
               --num_parts 100 \
               --cluster_batch_size 1 \
               --saint_num_steps 5 \
               --test_num_parts 10
```
## How to Run
First, ensure you have installed the required environments as mentioned above. You can then clone this repository and navigate to the directory where the main.py is located. After that, you can run the training process with the desired parameters.
