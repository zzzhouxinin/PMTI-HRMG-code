#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:59:06 2025

@author: zhouxin
"""


import time
import argparse
import pandas as pd
from utils import *

def set_args():
    parser = argparse.ArgumentParser(description='Graph construct by genome and miRNA information')

    parser.add_argument('--prefix', type=str,
                        default='Arabidopsis_thaliana',
                        help='The output file prefix')
    parser.add_argument('--miRNA_fa', type=str,
                        default='./data/Arabidopsis_thaliana/PmiREN/Arabidopsis_thaliana_mature.fa',
                        help='The mature miRNA sequence fasta file')
    parser.add_argument('--target', type=str,
                        default='./data/Arabidopsis_thaliana/PmiREN/Arabidopsis_thaliana_targetGene.txt',
                        help='The miRNA and target-gene information file')
    parser.add_argument('--transcript', type=str,
                        default='./data/Arabidopsis_thaliana/Genome/Athaliana_167_TAIR10.transcript.fa',
                        help='The transcript sequence fasta file')
    parser.add_argument('--kmer', type=int,
                        default=7,
                        help='The kmer length')

    args = parser.parse_args()

    return args

def localtime():
    return(time.strftime(' %Y-%m-%d %H:%M:%S\n --> ', time.localtime()))

if __name__ == '__main__':
    args = set_args()
    args_dict = vars(args)
    args_command = 'python graphconstruct.py'
    for i in args_dict.keys():
        args_command = args_command + ' --' + i
        args_command = args_command + ' ' + str(args_dict[i])

    print('Command: {}'.format(args_command))

    print('\033[1;36m{}\033[0m'.format(localtime()),'Initial parameters ...')
    print('\033[1;36m -->  miRNA:\033[0m {}'.format(args.miRNA_fa))
    print('\033[1;36m -->  target:\033[0m {}'.format(args.target))
    print('\033[1;36m -->  transcript:\033[0m {}\n'.format(args.transcript))

    # 1. Read fasta format to dict
    print('\033[1;36m{}\033[0m'.format(localtime()),'Read fasta format to dict ...\n')
    miRNA_fa = read_fasta_to_dict(args.miRNA_fa)
    trans_fa = read_fasta_to_dict(args.transcript)

    # 2. Obtain miRNA complement and reverse sequence
    print('\033[1;36m{}\033[0m'.format(localtime()),'Obtain miRNA complement and reverse sequence ...\n')
    miRNA_fa_com = complement_fasta_dict(miRNA_fa)
    miRNA_fa_rev = reverse_fasta_dict(miRNA_fa)
    miRNA_fa_rev_com = reverse_complement_fasta_dict(miRNA_fa)

    # 3. Stat kmer
    print('\033[1;36m{}\033[0m'.format(localtime()),'3. Stat kmer ...\n')
    miRNA_raw_kmer = stat_kmer(miRNA_fa, args.kmer)
    miRNA_com_kmer = stat_kmer(miRNA_fa_com, args.kmer)
    miRNA_rev_kmer = stat_kmer(miRNA_fa_rev, args.kmer)
    miRNA_rev_com_kmer = stat_kmer(miRNA_fa_rev_com, args.kmer)
    miRNA_kmer = (miRNA_com_kmer + miRNA_rev_com_kmer)

    trans_kmer = stat_kmer(trans_fa, args.kmer)

    miRNA_raw_kmer.to_csv('./process_data/{}_miRNA_raw_kmer.csv'.format(args.prefix), header=True, index=True)
    miRNA_com_kmer.to_csv('./process_data/{}_miRNA_com_kmer.csv'.format(args.prefix), header=True, index=True)
    miRNA_rev_kmer.to_csv('./process_data/{}_miRNA_rev_kmer.csv'.format(args.prefix), header=True, index=True)
    miRNA_rev_com_kmer.to_csv('./process_data/{}_miRNA_rev_com_kmer.csv'.format(args.prefix), header=True, index=True)
    miRNA_kmer.to_csv('./process_data/{}_miRNA_kmer.csv'.format(args.prefix), header=True, index=True)
    trans_kmer.to_csv('./process_data/{}_trans_kmer.csv'.format(args.prefix), header=True, index=True)

    # 4. Calculate cosine similarity
    print('\033[1;36m{}\033[0m'.format(localtime()),'Cosine similarity ...\n')
    miRNA_cosine = seq_cosine_similarity(miRNA_kmer)
    trans_cosine = seq_cosine_similarity(trans_kmer)

    miRNA_cosine.to_csv('./process_data/{}_miRNA_cosine_sim.csv'.format(args.prefix), header=True, index=True, float_format='%.4f')
    trans_cosine.to_csv('./process_data/{}_trans_cosine_sim.csv'.format(args.prefix), header=True, index=True, float_format='%.4f')

    # 5. Calculate Jaccard similarity
    print('\033[1;36m{}\033[0m'.format(localtime()),'Jaccard similarity ...\n')
    miRNA_jaccard = seq_jaccard_similarity(miRNA_kmer)
    trans_jaccard = seq_jaccard_similarity(trans_kmer)
    miRNA_jaccard.to_csv('./process_data/{}_miRNA_jaccard_sim.csv'.format(args.prefix), header=True, index=True, float_format='%.4f')
    trans_jaccard.to_csv('./process_data/{}_trans_jaccard_sim.csv'.format(args.prefix), header=True, index=True, float_format='%.4f')

    # 6. Construct heterogeneous graph
    #  -----------
    # | G-G | G-M |
    # |-----|-----|
    # | M-G | M-M |
    #  -----------
    # G=gene; M=miRNA
    print('\033[1;36m{}\033[0m'.format(localtime()),'Constructing graph ...\n')

    G_G = trans_cosine
    M_M = miRNA_cosine
    miRNA_target = pd.read_csv(args.target, sep='\t', header=0)
    miRNA_target['Value'] = 1
    miRNA_target_df = miRNA_target.pivot(index='target gene ID', columns='miRNA locus ID', values='Value')
    miRNA_target_df = miRNA_target_df.fillna(0)
    miRNA_target_df.columns.name = None
    miRNA_target_df.index.name = None
    G_M = pd.DataFrame(0, index = G_G.index.tolist(), columns=M_M.index.tolist())
    G_M = G_M.add(miRNA_target_df, fill_value=0)
    G_M = G_M.loc[G_G.index.tolist(), M_M.index.tolist()]
    M_G = G_M.T
    GG_GM = pd.concat([G_G, G_M], axis=1)
    MG_MM = pd.concat([M_G, M_M], axis=1)
    hete_graph = pd.concat([GG_GM, MG_MM], axis=0)
    G_G.to_csv('./process_data/{}_G_G_cosine_graph.csv'.format(args.prefix), header=True, index=True)
    G_M.to_csv('./process_data/{}_G_M_cosine_graph.csv'.format(args.prefix), header=True, index=True)
    M_G.to_csv('./process_data/{}_M_G_cosine_graph.csv'.format(args.prefix), header=True, index=True)
    M_M.to_csv('./process_data/{}_M_M_cosine_graph.csv'.format(args.prefix), header=True, index=True)
    hete_graph.to_csv('./process_data/{}_cosine_Heterogeneous_graph.csv'.format(args.prefix), header=True, index=True, float_format='%.4f')

    G_G = trans_jaccard
    M_M = miRNA_jaccard
    GG_GM = pd.concat([G_G, G_M], axis=1)
    MG_MM = pd.concat([M_G, M_M], axis=1)
    hete_graph = pd.concat([GG_GM, MG_MM], axis=0)
    G_G.to_csv('./process_data/{}_G_G_jaccard_graph.csv'.format(args.prefix), header=True, index=True)
    G_M.to_csv('./process_data/{}_G_M_jaccard_graph.csv'.format(args.prefix), header=True, index=True)
    M_G.to_csv('./process_data/{}_M_G_jaccard_graph.csv'.format(args.prefix), header=True, index=True)
    M_M.to_csv('./process_data/{}_M_M_jaccard_graph.csv'.format(args.prefix), header=True, index=True)
    hete_graph.to_csv('./process_data/{}_jaccard_Heterogeneous_graph.csv'.format(args.prefix), header=True, index=True, float_format='%.4f')

    # 7. Stat long kmer for node attr
    print('\033[1;36m{}\033[0m'.format(localtime()), 'Get node attributes ...\n')
    node_attr = stat_miRNA_in_gene(miRNA_com_dict=miRNA_fa_com, miRNA_rev_com_dict=miRNA_fa_rev_com, gene_dict=trans_fa, match_length=15)
    node_attr_order = node_attr.loc[hete_graph.index.tolist(),:]
    node_attr_order.to_csv('./process_data/{}_node_attr.csv'.format(args.prefix), header=True, index=True)

    node_attr_order_concat = pd.concat([node_attr_order, hete_graph], axis=1)
    node_attr_order_concat.to_csv('./process_data/{}_node_attr_jaccard.csv'.format(args.prefix), header=True, index=True)
