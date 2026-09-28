#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def read_fasta_to_dict(fasta_file: str = 'demo.fa') -> dict:
    """
    Read fasta file into a Python dictionary.

    Sequence names are used as dictionary keys and sequences as values.

    Parameters
    ----------
    fasta_file : str
        Fasta format file path.

    Returns
    -------
    dict
        A dictionary where keys are sequence names and values are sequences.
    """
    file = open(fasta_file, 'r')
    seq_dict = {}
    for i in file:
        i = i.strip()
        if len(i) > 0:
            if i[0] == '>':
                i = i.split(' ')
                key = i[0][1:]
                seq_dict[key] = ''
            else:
                seq_dict[key] += i

    return seq_dict


def complement_fasta_dict(fasta_dict: dict) -> dict:
    """
    Obtain complement sequences.

    Matching rules can be customized.

    Parameters
    ----------
    fasta_dict : dict
        Output from read_fasta_to_dict, keys are seq names, values are sequences.

    Returns
    -------
    dict
        Keys are sequence names, values are complement sequences.
    """
    transtab = str.maketrans('ATCG','TAGC')

    comp_dict = fasta_dict.copy()
    for i in comp_dict:
        comp_dict[i] = comp_dict[i].translate(transtab)

    return comp_dict


def reverse_fasta_dict(fasta_dict: dict) -> dict:
    """
    Obtain reverse sequences.

    Matching rules can be customized.

    Parameters
    ----------
    fasta_dict : dict
        Output from read_fasta_to_dict, keys are seq names, values are sequences.

    Returns
    -------
    dict
        Keys are sequence names, values are reverse sequences.
    """
    rev_dict = fasta_dict.copy()
    for i in rev_dict:
        rev_dict[i] = rev_dict[i][::-1]

    return rev_dict


def reverse_complement_fasta_dict(fasta_dict: dict) -> dict:
    """
    Obtain reverse complement sequences.

    Matching rules can be customized.

    Parameters
    ----------
    fasta_dict : dict
        Output from read_fasta_to_dict, keys are seq names, values are sequences.

    Returns
    -------
    dict
        Keys are sequence names, values are reverse complement sequences.
    """
    transtab = str.maketrans('ATCG','TAGC')

    rev_comp_dict = fasta_dict.copy()
    for i in rev_comp_dict:
        rev_comp_dict[i] = rev_comp_dict[i].translate(transtab)
        rev_comp_dict[i] = rev_comp_dict[i][::-1]

    return rev_comp_dict


def stat_kmer(seq_dict: dict, kmer_num: int) -> pd.DataFrame:
    """
    Compute sequence K-mer frequency.

    Rows represent sequences, columns represent kmer types.

    Parameters
    ----------
    seq_dict : dict
        Sequence dictionary, output from fasta_to_dict.py.

    kmer_num : int
        The K-mer length 'K'.

    Returns
    -------
    pd.DataFrame
        Rows are sequence names, columns are kmers.
        Values are kmer frequency counts in each sequence.
    """
    # Four nucleotide types
    nt_list = ['A','T','C','G']

    # Generate all kmer combinations
    from itertools import product
    kmer_list = list(product(nt_list, repeat=kmer_num))
    kmer_list = list(map(lambda x:''.join(x), kmer_list))

    # Define a DataFrame for storing results
    kmer_dt = pd.DataFrame(0, index=seq_dict.keys(), columns=kmer_list)

    for i in seq_dict:
        this_seq_name = i
        this_seq_seq = seq_dict[i]
        for j in range(len(this_seq_seq)-kmer_num+1):
            this_kmer = this_seq_seq[j:j+kmer_num]
            if this_kmer in kmer_list:
                kmer_dt.loc[this_seq_name, this_kmer] += 1

    return kmer_dt


def seq_cosine_similarity(kmer_dt: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sequence cosine similarity.

    Parameters
    ----------
    kmer_dt : pd.DataFrame
        DataFrame of sequence kmer frequencies, output from stat_kmer.py.

    Returns
    -------
    pd.DataFrame
        Rows and columns are sequence names, values are pairwise similarities.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cosine_value = cosine_similarity(kmer_dt.values, kmer_dt.values)
    cosine_similarity = pd.DataFrame(cosine_value,
                                     index=kmer_dt.index.tolist(),
                                     columns=kmer_dt.index.tolist())

    return cosine_similarity


def seq_jaccard_similarity(kmer_dt: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sequence Jaccard similarity.

    Parameters
    ----------
    kmer_dt : pd.DataFrame
        DataFrame of sequence kmer frequencies, output from stat_kmer.py.

    Returns
    -------
    pd.DataFrame
        Rows and columns are sequence names, values are pairwise similarities.
    """
    from sklearn.metrics import pairwise_distances

    # 'kmer_dt.values' > 0 to avoid warnings,
    # 'kmer_dt.values' also works
    jaccard_value = pairwise_distances(kmer_dt.values > 0, kmer_dt.values > 0,
                                       metric='jaccard')
    jaccard_similarity = pd.DataFrame(1 - jaccard_value,
                                     index=kmer_dt.index.tolist(),
                                     columns=kmer_dt.index.tolist())

    return jaccard_similarity


def stat_miRNA_in_gene(miRNA_com_dict: dict,
                          miRNA_rev_com_dict: dict,
                          gene_dict: dict,
                          match_length:int) -> pd.DataFrame:
    """
    Match miRNA with gene seed regions based on base complementarity.

    The goal of this function is to detect whether a mature miRNA could match
    a gene by the base complement rule. For example, setting match_length to 15
    will detect the frequency of this 15-base sequence in both miRNA and gene.
    If both miRNA and gene contain the 15-base sequence, they are considered
    matchable.

    Parameters
    ----------
    miRNA_com_dict : dict
        The miRNA complement dictionary.
    miRNA_rev_com_dict : dict
        The miRNA reverse complement dictionary.
    gene_dict : dict
        The gene dictionary.
    match_length : int
        Should not exceed the miRNA sequence length.

    Returns
    -------
    pd.DataFrame
        Rows are genes and miRNAs, columns are seed sequence types.
    """
    # Collect all seed types
    seed_list = []
    for i in miRNA_com_dict:
        com_seq = miRNA_com_dict[i]
        rc_seq = miRNA_rev_com_dict[i]
        for j in range(len(com_seq)-match_length+1):
            com_seed = com_seq[j:j+match_length]
            if com_seed not in seed_list:
                seed_list.append(com_seed)

            rc_seed = rc_seq[j:j+match_length]
            if rc_seed not in seed_list:
                seed_list.append(rc_seed)

    # Detect whether a seed exists in miRNA and gene sequences
    node_list = [*gene_dict] + [*miRNA_com_dict]
    seed_num = pd.DataFrame(0, index=node_list, columns=seed_list)

    # For gene nodes
    for i in [*gene_dict]:
        for j in seed_list:
            if j in gene_dict[i]:
                seed_num.loc[i,j] = gene_dict[i].count(j)

    # For miRNA nodes
    for i in [*miRNA_com_dict]:
        for j in seed_list:
            if (j in miRNA_com_dict[i]) or (j in miRNA_rev_com_dict[i]):
                seed_num.loc[i,j] = miRNA_com_dict[i].count(j) + miRNA_rev_com_dict[i].count(j)

    return seed_num
