#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:59:06 2025

@author: zhouxin
"""

import time
import torch
import argparse
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import TruncatedSVD

from model.DAE import DAE


def set_args():
    parser = argparse.ArgumentParser(
        description='Sequence + Topology Feature Fusion Encoding'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='Arabidopsis_thaliana',
        help='The output file prefix'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='cuda/cpu'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )

    parser.add_argument(
        '--epoch',
        type=int,
        default=1000,
        help='Training epochs'
    )

    parser.add_argument(
        '--D',
        type=int,
        default=256,
        help='Sequence embedding dimension'
    )

    parser.add_argument(
        '--graph_dim',
        type=int,
        default=64,
        help='Topology embedding dimension (SVD)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Batch size'
    )

    return parser.parse_args()


def localtime():
    return time.strftime(
        ' %Y-%m-%d %H:%M:%S\n --> ',
        time.localtime()
    )


if __name__ == '__main__':

    args = set_args()

    args_dict = vars(args)

    args_command = 'python feture_encoded.py'

    for k in args_dict.keys():
        args_command += f' --{k} {args_dict[k]}'

    print('Command: {}'.format(args_command))

    # =====================================================
    # 1. Load Sequence Features
    # =====================================================

    print(
        '\033[1;36m{}\033[0m'.format(localtime()),
        'Import node attribute data ...\n'
    )

    node_attr = pd.read_csv(
        './process_data/{}_node_attr.csv'.format(args.prefix),
        header=0,
        index_col=0
    )

    node_attr_tensor = torch.tensor(
        node_attr.values,
        dtype=torch.float32
    )

    dataset = TensorDataset(node_attr_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # =====================================================
    # 2. DAE Training
    # =====================================================

    print(
        '\033[1;36m{}\033[0m'.format(localtime()),
        'Start DAE training ...\n'
    )

    model = DAE(
        node_attr_tensor.shape[1],
        args.D
    ).to(args.device)

    criterion = torch.nn.MSELoss(
        reduction='sum'
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    for epoch in range(args.epoch):

        model.train()

        total_loss = 0.0

        for batch in dataloader:

            batch_x = batch[0].to(args.device)

            optimizer.zero_grad()

            encoded, decoded = model(batch_x)

            loss = criterion(
                decoded,
                batch_x
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:

            print(
                '\033[5;33m'
                'DAE Epoch:[{}/{}], Loss:{:.5f}'
                '\033[0m'.format(
                    epoch + 1,
                    args.epoch,
                    total_loss
                )
            )

    # =====================================================
    # 3. Extract Sequence Embedding
    # =====================================================

    print(
        '\033[1;36m{}\033[0m'.format(localtime()),
        'Extract sequence embedding ...\n'
    )

    model.eval()

    encoded_outputs = []

    with torch.no_grad():

        inference_loader = DataLoader(
            TensorDataset(node_attr_tensor),
            batch_size=args.batch_size,
            shuffle=False
        )

        for batch in inference_loader:

            batch_x = batch[0].to(args.device)

            batch_encoded, _ = model(batch_x)

            encoded_outputs.append(
                batch_encoded.cpu()
            )

    seq_embedding = torch.cat(
        encoded_outputs,
        dim=0
    ).numpy()

    print(
        f'Sequence Embedding Shape: {seq_embedding.shape}'
    )

    # =====================================================
    # 4. Load Heterogeneous Graph
    # =====================================================

    print(
        '\033[1;36m{}\033[0m'.format(localtime()),
        'Import heterogeneous graph ...\n'
    )

    heter_graph = pd.read_csv(
        './process_data/{}_jaccard_Heterogeneous_graph.csv'.format(
            args.prefix
        ),
        header=0,
        index_col=0
    )

    # =====================================================
    # 5. SVD Topology Embedding
    # =====================================================

    print(
        '\033[1;36m{}\033[0m'.format(localtime()),
        'Topology embedding by SVD ...\n'
    )

    svd = TruncatedSVD(
        n_components=args.graph_dim,
        random_state=42
    )

    graph_embedding = svd.fit_transform(
        heter_graph.values
    )

    explained_ratio = svd.explained_variance_ratio_.sum()

    print(
        f'Topology Embedding Shape: {graph_embedding.shape}'
    )

    print(
        f'Cumulative Explained Variance Ratio: '
        f'{explained_ratio:.4f}'
    )

    # =====================================================
    # 6. Fusion
    # =====================================================

    print(
        '\033[1;36m{}\033[0m'.format(localtime()),
        'Feature fusion ...\n'
    )

    fusion_feature = np.concatenate(
        [
            seq_embedding,
            graph_embedding
        ],
        axis=1
    )

    print(
        f'Fusion Feature Shape: {fusion_feature.shape}'
    )

    # =====================================================
    # 7. Save
    # =====================================================

    fusion_df = pd.DataFrame(
        fusion_feature,
        index=node_attr.index.tolist()
    )

    output_path = (
        './process_data/{}_node_attr_encode.csv'
        .format(args.prefix)
    )

    fusion_df.to_csv(
        output_path,
        header=True,
        index=True,
        float_format='%.10f'
    )

    print(
        '\033[1;32m{}\033[0m'.format(localtime()),
        f'Saved to: {output_path}'
    )