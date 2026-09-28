#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:59:06 2025

@author: zhouxin
"""

import os
import gc
import time
import torch
import random
import argparse
import pandas as pd
import numpy as np
from model.PygLinkPrediction import PygLinkPrediction
from model.MLP import MLP
from model.MakePygFormat import MakePygFormat
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from operator import itemgetter
from sklearn.metrics import precision_recall_curve,roc_curve,auc,precision_score,accuracy_score,recall_score,f1_score

def set_args():
    parser = argparse.ArgumentParser(description='End-to-end training')
    
    parser.add_argument('--method', type=str, choices=['HRMG','GCN','ChebConv','GAT'],
                        default='HRMG',
                        help='The graph convolutional method')
    parser.add_argument('--prefix', type=str,
                        default='Arabidopsis_thaliana',
                        help='The output file prefix, MUST same as step1')
    parser.add_argument('--device', type=str,
                        default='cpu',
                        help='The device, cuda/cpu')
    parser.add_argument('--lr', type=float,
                        default=0.0001,
                        help='Learning rate')
    parser.add_argument('--epoch', type=int,
                        default=3000,
                        help='Number of training epochs')
    parser.add_argument('--kfold', type=int,
                        default=5,
                        help='Number of cross validation')
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help='Threshold of edges')
    parser.add_argument('--graphdim1', type=int,
                        default=512,
                        help='Dimensions of graph layer 1')
    parser.add_argument('--graphdim2', type=int,
                        default=256,
                        help='Dimensions of graph layer 2')
    parser.add_argument('--graphdim3', type=int,
                        default=128,
                        help='Dimensions of graph layer 3 and MLP layer 1')
    parser.add_argument('--mlpdim2', type=int,
                        default=256,
                        help='Dimensions of MLP layer 2')
    parser.add_argument('--mlpdim3', type=int,
                        default=64,
                        help='Dimensions of MLP layer 3')
    parser.add_argument('--chebconvk', type=int,
                        default=1,
                        help='Number of ChebConv K')
    parser.add_argument('--heads', type=int,
                        default=1,
                        help='Number of multi-head-attentions')
    parser.add_argument('--simthreshold', type=float,
                        default=0.4,
                        help='Threhold of similarity')
    parser.add_argument('--D', type=int,
                        default=256,
                        help='Recommended to try 16/32/64/128/256/512/1024, higher D corresponds to slower speeds and exponential memory consumption')
    parser.add_argument('--sim', type=str,
                        default='jaccard',
                        help='The similarity method, [cosine/jaccard]')
    
    args = parser.parse_args()
    
    return args

def localtime():
    return(time.strftime(' %Y-%m-%d %H:%M:%S\n --> ', time.localtime()))

def seed_torch(seed: int = 123) -> None:
    # Set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    print(f"Random seed set as {seed}")
    
def load_graph_model(method):
    if method == 'GCN':
        from model.GraphConvolutional import GraphConvolutional
        graph_model = GraphConvolutional(node_attr=x,
                                         dim_l1=args.graphdim1,
                                         dim_l2=args.graphdim2,
                                         dim_output=args.graphdim3).to(args.device)
    elif method == 'ChebConv':
        from model.ChebConvConvolutional import ChebConvConvolutional
        graph_model = ChebConvConvolutional(node_attr=x,
                                            ChebConvK=args.chebconvk,
                                            dim_l1=args.graphdim1,
                                            dim_l2=args.graphdim2,
                                            dim_output=args.graphdim3).to(args.device)
    elif method == 'GAT':
        from model.GAT import GAT
        graph_model = GraphAttentional(node_attr=x,
                                       dim_l1=args.graphdim1,
                                       dim_l2=args.graphdim2,
                                       dim_output=args.graphdim3,
                                       heads=args.heads).to(args.device)
        
    elif method == 'HRMG':
        from model.HRMG import HRMG
        graph_model = HRMG(in_dim=x.shape[1],  
                            hidden_dims=(args.graphdim3, args.graphdim3),
                            out_dim=args.graphdim3).to(args.device)

    return graph_model

def plot_attention_heatmap(attn_weights, save_path, max_nodes=50):

    attn = attn_weights.detach().cpu().numpy()

    # Truncate if too many nodes to prevent memory overflow
    if attn.shape[0] > max_nodes:
        attn = attn[:max_nodes]

    plt.figure(figsize=(6, 8))
    plt.imshow(attn, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Attention Weight')

    plt.xticks(
    ticks=[0, 1, 2],
    labels=['K=2', 'K=4', 'K=6']
    )

    plt.xlabel('Chebyshev Polynomial Order')
    plt.ylabel('Node Index')
    plt.title('Node-level Multi-scale Attention Heatmap')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    
    args = set_args()
    args_dict = vars(args)
    args_command = 'python model_train.py'
    for i in args_dict.keys():
        args_command = args_command + ' --' + i
        args_command = args_command + ' ' + str(args_dict[i])
    
    print('Command: {}'.format(args_command))
    seed_torch()
    
    # 1. Import data
    print('\033[1;36m{}\033[0m'.format(localtime()),'Import data ...\n')
    G_G = pd.read_csv('./process_data/{}_G_G_{}_graph.csv'.format(args.prefix, args.sim), header=0, index_col=0)
    M_M = pd.read_csv('./process_data/{}_M_M_{}_graph.csv'.format(args.prefix, args.sim), header=0, index_col=0)
    G_M = pd.read_csv('./process_data/{}_G_M_{}_graph.csv'.format(args.prefix, args.sim), header=0, index_col=0)
    M_G = G_M.T
    node_attr = pd.read_csv('./process_data/{}_D{}_node_attr_encode.csv'.format(args.prefix, args.D), header=0, index_col=0)
    
    # 2. Prapare node, edges
    print('\033[1;36m{}\033[0m'.format(localtime()),'Convert heterogeneous graph to edges table ...\n')
    G_G[G_G < args.simthreshold] = 0
    M_M[M_M < args.simthreshold] = 0

    gene_names = G_G.index.tolist()  # Gene node names
    mirna_names = M_M.index.tolist()  # miRNA node names

    # Create mapping from node names to indices
    all_node_names = gene_names + mirna_names
    all_node_name_index_dict = {name: idx for idx, name in enumerate(all_node_names)}

    # Apply threshold and convert to sparse edge representation
    def df_to_sparse_edges(df, threshold, source_offset=0, target_offset=0):
        """Convert DataFrame to sparse edge list"""
        # Get positions and values of non-zero elements
        rows, cols = np.where(df.values >= threshold)
        values = df.values[rows, cols]
        
        # Create edge list
        edges = pd.DataFrame({
            'From': rows + source_offset,
            'To': cols + target_offset,
            'Value': values
        })
        return edges

    # Get node count information
    n_genes = G_G.shape[0]
    n_mirnas = M_M.shape[0]

    # Process edges from four subgraphs separately
    edges_gg = df_to_sparse_edges(G_G, args.simthreshold, 0, 0)
    edges_mm = df_to_sparse_edges(M_M, args.simthreshold, n_genes, n_genes)
    edges_gm = df_to_sparse_edges(G_M, args.simthreshold, 0, n_genes)
    edges_mg = df_to_sparse_edges(M_G, args.simthreshold, n_genes, 0)

    # Merge all edges
    gcn_edges = pd.concat([edges_gg, edges_mm, edges_gm, edges_mg], ignore_index=True)
    gcn_edges = gcn_edges[gcn_edges['Value'] > 0].reset_index(drop=True)

    print('\033[1;36m{}\033[0m'.format(localtime()),'Totally {} edges used for graph convolution ...\n'.format(len(gcn_edges)))

    del G_G, M_M
    print(f"Memory @here: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.1f} MB")
    G_M_edges = G_M.melt(ignore_index=False).reset_index()
    G_M_edges.columns = ['From', 'To', 'Value']
    mlp_pos_edges = G_M_edges[G_M_edges['Value'] == 1]
    mlp_neg_edges = G_M_edges[G_M_edges['Value'] == 0]
    mlp_neg_edges = mlp_neg_edges[(mlp_neg_edges['From'].isin(mlp_pos_edges['From']))&(mlp_neg_edges['To'].isin(mlp_pos_edges['To']))]
    mlp_pos_edges = mlp_pos_edges.reset_index(drop=True)
    mlp_neg_edges = mlp_neg_edges.reset_index(drop=True)
    mlp_pos_edges_index = mlp_pos_edges.index.tolist()
    mlp_neg_edges_index = mlp_neg_edges.index.tolist()
    random.shuffle(mlp_pos_edges_index)
    random.shuffle(mlp_neg_edges_index)
    mlp_pos_edges = mlp_pos_edges.loc[mlp_pos_edges_index,].reset_index(drop=True)
    mlp_neg_edges = mlp_neg_edges.loc[mlp_neg_edges_index,].sample(len(mlp_pos_edges)).reset_index(drop=True)
    
    mlp_pos_edges['FromIndex'] = mlp_pos_edges['From'].map(all_node_name_index_dict)
    mlp_pos_edges['ToIndex'] = mlp_pos_edges['To'].map(all_node_name_index_dict)
    mlp_neg_edges['FromIndex'] = mlp_neg_edges['From'].map(all_node_name_index_dict)
    mlp_neg_edges['ToIndex'] = mlp_neg_edges['To'].map(all_node_name_index_dict)
    mlp_pos_edges['Label'] = 1
    mlp_neg_edges['Label'] = 0
    print('\033[1;36m{}\033[0m'.format(localtime()),'Totally {} positive edges used for link prediction ...\n'.format(len(mlp_pos_edges)))
    print('\033[1;36m{}\033[0m'.format(localtime()),'Totally {} negative edges used for link prediction ...\n'.format(len(mlp_neg_edges)))
    
    # 3. Split k fold
    k = 0
    kf = KFold(n_splits=args.kfold)
    final_auc = []
    final_aupr = []
    final_pre = []
    final_rec = []
    final_acc = []
    final_f1 = []
    final_loss = []
    
    train_pos_list = []
    test_pos_list = []
    train_neg_list = []
    test_neg_list = []

    all_fold_train_loss = []  # Store training loss for all epochs in each fold
    all_fold_test_loss = []   # Store test loss for each fold
    
    train_pos_list = []
    test_pos_list = []
    train_neg_list = []
    test_neg_list = []

    
    print('\033[1;36m{}\033[0m'.format(localtime()),'Start training ...\n')
    for train_pos, test_pos in kf.split(mlp_pos_edges):
        k += 1
        print('\033[1;31m     ========== Fold [{}/{}] on train dataset ==========\033[0m'.format(k, args.kfold))
        mlp_pos_edges_train = mlp_pos_edges.iloc[train_pos,].copy()
        mlp_neg_edges_train = mlp_neg_edges.iloc[train_pos,].copy()
        mlp_pos_edges_test = mlp_pos_edges.iloc[test_pos,].copy()
        mlp_neg_edges_test = mlp_neg_edges.iloc[test_pos,].copy()
        
        mlp_edges_train = pd.concat([mlp_pos_edges_train, mlp_neg_edges_train]).reset_index(drop=True)
        mlp_edges_test = pd.concat([mlp_pos_edges_test, mlp_neg_edges_test]).reset_index(drop=True)
        mlp_edges_train_label = torch.tensor(mlp_edges_train['Label'].tolist(), dtype=torch.float32).to(args.device)
        mlp_edges_test_label = torch.tensor(mlp_edges_test['Label'].tolist(), dtype=torch.float32).to(args.device)
        
        gcn_edges_train = gcn_edges.copy()
        gcn_edges_test = gcn_edges.copy()
        
        for i in range(len(mlp_edges_test)):
            a = mlp_edges_test['FromIndex'].tolist()[i]
            b = mlp_edges_test['ToIndex'].tolist()[i]
            gcn_edges_train.loc[(gcn_edges_train['From'] == a)&(gcn_edges_train['To'] == b), 'Value'] = 0
            gcn_edges_train.loc[(gcn_edges_train['From'] == b)&(gcn_edges_train['To'] == a), 'Value'] = 0
            gcn_edges_train.loc[(gcn_edges_train['From'] == a)&(gcn_edges_train['To'] == b), 'Zvalue'] = 0
            gcn_edges_train.loc[(gcn_edges_train['From'] == b)&(gcn_edges_train['To'] == a), 'Zvalue'] = 0
        gcn_edges_train = gcn_edges_train[gcn_edges_train['Value'] > 0]
        
        for i in range(len(mlp_edges_train)):
            a = mlp_edges_train['FromIndex'].tolist()[i]
            b = mlp_edges_train['ToIndex'].tolist()[i]
            gcn_edges_test.loc[(gcn_edges_test['From'] == a)&(gcn_edges_test['To'] == b), 'Value'] = 0
            gcn_edges_test.loc[(gcn_edges_test['From'] == b)&(gcn_edges_test['To'] == a), 'Value'] = 0
            gcn_edges_test.loc[(gcn_edges_test['From'] == a)&(gcn_edges_test['To'] == b), 'Zvalue'] = 0
            gcn_edges_test.loc[(gcn_edges_test['From'] == b)&(gcn_edges_test['To'] == a), 'Zvalue'] = 0
        gcn_edges_test = gcn_edges_test[gcn_edges_test['Value'] > 0]
        
        print('\033[1;31m     Totally {} edges for GCN\033[0m'.format(len(gcn_edges_train)))
        
        # Edge
        gcn_edges_train1 = np.array(list(zip(gcn_edges_train['From'], gcn_edges_train['To'])))
        gcn_edges_train2 = np.array(list(zip(gcn_edges_train['To'], gcn_edges_train['From'])))
        gcn_edges_train_idx = np.concatenate((gcn_edges_train1,gcn_edges_train2), axis=0)
        gcn_edges_train_idx = torch.tensor(gcn_edges_train_idx, dtype=torch.long).to(args.device)
        
        
        # Edge weight
        gcn_edges_train_weight = torch.tensor(gcn_edges_train['Value'].tolist(), dtype=torch.float32)
        gcn_edges_train_weight = torch.reshape(gcn_edges_train_weight, [len(gcn_edges_train_weight), 1])
        gcn_edges_train_weight = torch.tile(gcn_edges_train_weight, (2,1)).to(args.device)
        
        # Node attributes
        x = torch.tensor(node_attr.values, dtype=torch.float32).to(args.device)

        
        # Instantiating model
        graph_model = load_graph_model(method=args.method)
        mlp_model = MLP(dim_input=args.graphdim3*2, dim_l1=args.mlpdim2, dim_l2=args.mlpdim3).to(args.device)
        
        # Define criterion
        criterion = torch.nn.BCEWithLogitsLoss()
        # criterion = torch.nn.BCELoss()
        
        # Define optimizer
        optimizer = torch.optim.Adam([{'params': graph_model.parameters(), 'lr': args.lr},
                                      {'params': mlp_model.parameters(), 'lr': args.lr}])
        
        # List to store loss
        loss_list = []
        
        # All done, start training!
        graph_model.train()
        mlp_model.train()

        # Before training loop, initialize per-fold test loss list
        fold_test_loss_list = []  # Store test loss for each epoch in current fold

        for epoch in range(args.epoch):
            optimizer.zero_grad()
            with autocast(device_type='cuda:0'):
                if args.method == 'PMTISM1' or args.method == 'PMTISM4':
                    gcn_ebd, attn_weights = graph_model(x, gcn_edges_train_idx.T, gcn_edges_train_weight)
                else:
                    gcn_ebd = graph_model(x, gcn_edges_train_idx.T, gcn_edges_train_weight)
                # Concat link embedding
                mlp_edge_ebd = torch.cat((gcn_ebd[mlp_edges_train['FromIndex'].tolist()], gcn_ebd[mlp_edges_train['ToIndex'].tolist()]), 1)
                
                # MLP
                mlp_out = mlp_model(mlp_edge_ebd).squeeze()
                
                # Backward
                loss = criterion(mlp_out, mlp_edges_train_label.float())
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())

            if (epoch+1)%100 == 0:
                print('\033[5;33m     Method:[{}], Fold:[{}/{}], Epoch:[{}/{}], Loss:{:.5f}\033[0m'.format(args.method,k, args.kfold, epoch+1, args.epoch, loss.item()))


        # Testing
        graph_model.eval()
        mlp_model.eval()
        with torch.no_grad():
            del gcn_edges_train_idx
            del gcn_edges_train_weight
            torch.cuda.empty_cache()
            
            # Edge 
            gcn_edges_test1 = np.array(list(zip(gcn_edges_test['From'], gcn_edges_test['To'])))
            gcn_edges_test2 = np.array(list(zip(gcn_edges_test['To'], gcn_edges_test['From'])))
            gcn_edges_test_idx = np.concatenate((gcn_edges_test1,gcn_edges_test2), axis=0)
            gcn_edges_test_idx = torch.tensor(gcn_edges_test_idx, dtype=torch.long).to(args.device)
            
            # Edge weight
            gcn_edges_test_weight = torch.tensor(gcn_edges_test['Value'].tolist(), dtype=torch.float32)
            gcn_edges_test_weight = torch.reshape(gcn_edges_test_weight, [len(gcn_edges_test_weight), 1])
            gcn_edges_test_weight = torch.tile(gcn_edges_test_weight, (2,1)).to(args.device)
            
            if args.method == 'HRMG':
                gcn_ebd, attn_weights = graph_model(x, gcn_edges_test_idx.T, gcn_edges_test_weight)
            else:
                gcn_ebd = graph_model(x, gcn_edges_test_idx.T, gcn_edges_test_weight)
            
            # Concat link embedding
            mlp_edge_ebd = torch.cat((gcn_ebd[mlp_edges_test['FromIndex'].tolist()], gcn_ebd[mlp_edges_test['ToIndex'].tolist()]), 1)
            
            # MLP
            mlp_out = mlp_model(mlp_edge_ebd).squeeze()
            
            # Loss
            test_loss = criterion(mlp_out, mlp_edges_test_label.float())
            
            y_true = mlp_edges_test_label.cpu().numpy().flatten()
            y_scores = mlp_out.detach().cpu().numpy().flatten()
            
            test_output = pd.DataFrame()
            test_output['From'] = mlp_edges_test['From']
            test_output['To'] = mlp_edges_test['To']
            test_output['Label'] = mlp_edges_test['Label']
            test_output['Predict'] = y_scores
            test_output.to_csv("./train_result/{}_{}_K_{}_scores.csv".format(args.prefix, args.method, k), index=False, header=True)
            
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            pd.DataFrame(fpr).to_csv("./train_result/{}_{}_K_{}_fpr.csv".format(args.prefix, args.method, k), index=False, header=False)
            pd.DataFrame(tpr).to_csv("./train_result/{}_{}_K_{}_tpr.csv".format(args.prefix, args.method, k), index=False, header=False)
            pd.DataFrame(precision).to_csv("./train_result/{}_{}_K_{}_precision.csv".format(args.prefix, args.method, k), index=False, header=False)
            pd.DataFrame(recall).to_csv("./train_result/{}_{}_K_{}_recall.csv".format(args.prefix, args.method, k), index=False, header=False)
            pd.DataFrame(loss_list).to_csv("./train_result/{}_{}_K_{}_loss.csv".format(args.prefix, args.method, k), index=False, header=False)
            
            auroc_score = auc(fpr, tpr)
            auprc_score = auc(recall, precision)
            
            auroc_score = auc(fpr, tpr)
            auprc_score = auc(recall, precision)
            
            y_pred = y_scores.copy()
            y_pred[y_pred >= args.threshold] = 1
            y_pred[y_pred < args.threshold] = 0
            
            acc = accuracy_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            print('\033[1;31m     ========== Fold [{}/{}] on test dataset ==========\033[0m'.format(k, args.kfold))
            print('\033[1;31m     METHOD:{}, LOSS:{:.5f}, AUC:{:.5f}, AUPR:{:.5f}, PRE:{:.5f}, REC:{:.5f}, ACC:{:.5f}, F1:{:.5f}\n\033[0m'.format(args.method, test_loss, auroc_score, auprc_score, pre, rec, acc, f1))
            final_auc.append(auroc_score)
            final_aupr.append(auprc_score)
            final_pre.append(pre)
            final_rec.append(rec)
            final_acc.append(acc)
            final_f1.append(f1)
            final_loss.append(test_loss.cpu())

            # ===== Plot multi-scale attention heatmap =====
            if args.method == 'HRMG' :
                heatmap_path = (
                    f"./picture_results/{args.prefix}_{args.method}_fold{k}_attention_heatmap.png"
                )
                plot_attention_heatmap(attn_weights, heatmap_path)
                print(f"Attention heatmap saved to: {heatmap_path}")
            
            # Save model
            torch.save(graph_model.state_dict(), "./train_result/{}_{}_K_{}_GRAPH.pt".format(args.prefix, args.method, k))
            torch.save(mlp_model.state_dict(), "./train_result/{}_{}_K_{}_MLP.pt".format(args.prefix, args.method, k))

    print('\033[1;36m     ********** Overview of {}-fold CV results of {} **********\033[0m'.format(args.kfold, args.method))
    for i in range(args.kfold):
        print('\033[1;36m     {} K[{}/{}] - LOSS:{:.5f}, AUC:{:.5f}, AUPR:{:.5f}, PRE:{:.5f}, REC:{:.5f}, ACC:{:.5f}, F1:{:.5f}\n\033[0m'.format(args.method, i+1, args.kfold, final_loss[i], final_auc[i], final_aupr[i], final_pre[i], final_rec[i], final_acc[i], final_f1[i]))
        
    print('\033[1;36m    {} Average - LOSS:{:.5f}, AUC:{:.5f}, AUPR:{:.5f}, PRE:{:.5f}, REC:{:.5f}, ACC:{:.5f}, F1:{:.5f}\n\033[0m'.format(args.method, np.mean(final_loss), np.mean(final_auc), np.mean(final_aupr), np.mean(final_pre), np.mean(final_rec), np.mean(final_acc), np.mean(final_f1)))
            
