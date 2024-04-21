#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import sys
import argparse
import subprocess as sp
from multiprocessing import Pool
from itertools import combinations
import pandas as pd
import numpy as np
from scipy import stats



def calc_MR(rank1, rank2):
    MR = np.sqrt(rank1 * rank2)
    return MR


def get_MR_from_correlations(correlations_df, all_objects):
    # GET RANKS
    print('Getting ranks...')
    ranks_dict = {}
    for cur_object in all_objects:
        df1 = correlations_df[correlations_df.metabolite == cur_object]
        df1 = df1[['gene', 'correlation']]
        df1.columns = ['other_object', 'correlation']

        df2 = correlations_df[correlations_df.gene == cur_object]
        df2 = df2[['metabolite', 'correlation']]
        df2.columns = ['other_object', 'correlation']

        dff = pd.concat([df1, df2]).sort_values(by='correlation', ascending=False)
        dff.index = np.arange(1, len(dff) + 1)  # this makes index = rank(cur_gene -> other_gene)

        ranks_dict[cur_object] = {}
        for rank, cur_row in dff.iterrows():
            ranks_dict[cur_object][cur_row.other_object] = rank

    # GET MUTUAL RANKS
    print('Getting mutual ranks...')
    edges_df = []
    object_combinations = {n for n in combinations(all_objects, 2)}
    for metabolite, gene in object_combinations:
        if ((metabolite in ranks_dict and gene in ranks_dict and gene in ranks_dict[metabolite] and metabolite in ranks_dict[gene])):
            rank1 = ranks_dict[metabolite][gene]
            rank2 = ranks_dict[gene][metabolite]
            edges_df.append([metabolite, gene, calc_MR(rank1, rank2)])
    edges_df = pd.DataFrame.from_records(edges_df, columns=['metabolite', 'gene', 'MR'])

    # OUTPUT
    #fname = os.path.join(Options.output_folder, 'MR.csv')
    #edges_df.to_csv(fname, index=False)
    return edges_df

def get_weight_from_MR(edges_df, edge_weight_cutoff, decay_rate):
    # CONVERT TO EDGE WEIGHT
    print('Estimating edge weights using exponetial decay...')
    edges_df['weight'] = np.exp(-(edges_df.MR - 1) / decay_rate)  # other options: /5 /10 /25 /50 /100
    weights_df = edges_df[edges_df.weight > edge_weight_cutoff].reset_index(drop=True) # default edge_weight_cutoff is 0.01
    
    return weights_df

def run_clusterone(outfile, clusterone, weights_file):
    # CLUSTERONE
    print('\n')
    with open(outfile, 'w') as f:
        # java -jar cluster_one-1.0.jar input -F csv
        sp.run(['java', '-jar', clusterone, weights_file, '-F', 'csv'], stdout=f)
    print('\n')

    return

