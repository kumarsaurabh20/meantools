#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from multiprocessing import Pool, Manager
import sqlite3
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from tqdm import tqdm

# project specific module
import mutual_ranks
import gizmos

def get_args():
    parser = argparse.ArgumentParser(prog='corrMultiomics',
                                     description='Correlate multi-omics and convert correlation coeffecient into edge weights!',
                                     epilog='Contact - kumarsaurabh.singh@maastrichtuniversity.nl')
    
    required = parser.add_argument_group('Required arguments')
    required.add_argument('-ft','--feature_table', default='', action='store', required=True, help='Metabolomics-based feature table.')
    required.add_argument('-qm','--quantitation_matrix', default='', action='store', required=False, help='Normalized expression table from the RNAseq data.')
    required.add_argument('-dn', '--sqlite_db_name', default='', action='store', required=True, help='Provide a name for the database!')
    required.add_argument('-tn', '--sqlite_table_name', default='', action='store', required=True, help='Provide a name for the database table.')

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('-m', '--method', default='pearson', type=str, required=False, choices=['spearman', 'pearson', 'pearsonlog'], help='Default is the pearson correlation method; pearsonlog uses log-transformed arrays, where arrays with atleast a zero are always removed first regardless of other options.')
    optional.add_argument('-mr', '--mutual_rank', default=False, required=False, action='store_true', help='MR has more module predictive power than pearson or pearson log.')
    optional.add_argument('-cl', '--clusterone', default=False, required=False, action='store_true', help='Clusterone creates modules based on MR and edge weights.')
    optional.add_argument('-mad', '--mad_filter', default=False, required=False, action='store_true', help='Removes arrays with a MAD of 0.')
    optional.add_argument('-r', '--remove_zeros', default=False, required=False, action='store_true', help='Removes arrays with at least one 0.')
    optional.add_argument('-c', '--correlation_cutoff', default=0.1, required=False, type=float, help='Minimum correlation coeffecient cut-off. Default: 0.3')
    optional.add_argument('-w', '--edge_weight_cutoff', default=0.01, required=False, type=float, help='Minimum MR-derived edge weight. Default: 0.01')
    optional.add_argument('-d', '--decay_rate', default=5, required=False, type=int, help='Decay rate of default: 25 would be used.')
    optional.add_argument('-mdr', '--multi_decay_rates', default=[5,10,25], required=False, nargs='+', type=int, help='Decay rate could be any value but generally it is either {5, 10, 25, 50, 100}. In this case all 5 values will be used separately to estimate edges. Usage:: -mdr 5 10 25 50 100')
    optional.add_argument('-a', '--annotation', default='', type=str, required=False, help='Comma-delimited two-columns file with annotations. No header.')
    optional.add_argument('-p', '--plot', default=False, required=False, action='store_true', help='Plots showing the correlation will be created.')
    optional.add_argument('-t', '--threads', default=8, type=int, required=False)
    optional.add_argument('-v','--verbose', default=False, action='store_true', required=False)
    return parser.parse_args()

def calc_corr(transcript_series, metabolite_series, Options):
    method = Options.method
    correlation = 0
    P = 0
    if method == 'pearson':
        correlation, P = stats.pearsonr(transcript_series, metabolite_series)
    elif method == 'pearsonlog':
        correlation, P = stats.pearsonr(np.log10(transcript_series), np.log10(metabolite_series))
    elif method == 'spearman':
        correlation, P = stats.spearmanr(transcript_series, metabolite_series)
    transcript_series['correlation'] = correlation
    transcript_series['P'] = P
    return transcript_series


def write_and_plot_output(results):
    results_df = results
    # WRITE OUTPUT
    # metabolite, gene, correlation, P, gene_annotation
    output_file = "results.csv"

    with open(output_file, 'a') as f:
        for i, cur_row in results_df.iterrows():
            f.write(cur_row.metabolite + ',')
            f.write(cur_row.gene + ',')
            f.write(str(cur_row.correlation) + ',')
            f.write(str(cur_row.P) + "\n")
    return


def corr_w_transcriptome(metabolite, transcripts_df, Options):
    cur_metabolite_transcripts_df = transcripts_df.apply(calc_corr, args=(metabolite, Options,), axis=1)
    cutoff_mask = abs(cur_metabolite_transcripts_df.correlation) >= Options.correlation_cutoff
    cur_metabolite_transcripts_df = cur_metabolite_transcripts_df[cutoff_mask]
    cur_metabolite_transcripts_df = cur_metabolite_transcripts_df.sort_values(by='correlation', ascending=False)
    results_df = pd.DataFrame({'metabolite': metabolite.name, 'gene': cur_metabolite_transcripts_df.index, 'correlation': cur_metabolite_transcripts_df.correlation, 'P': cur_metabolite_transcripts_df.P})
    return results_df


def print_child_proc_error(error_string):
    print('Child process encountered the following error: ' + str(error_string))
    return

def calc_MAD(x):
    """
    Returns the median absolute deviation via the following equation:
    MAD = median( | Xi - median(X) |  )
    :param x:
    :return:
    """
    med = np.median(x)
    x2 = np.absolute(x-med)
    MAD = np.median(x2)
    return MAD


def apply_MAD_filter(df):
    df['MAD'] = df.apply(calc_MAD, axis=1)
    mask = df.MAD > 0
    df_filtered = df[mask]
    del df_filtered['MAD']
    return df_filtered



def main(Options):

    script = os.path.dirname(os.path.abspath(__file__))
    pool = Pool(processes=Options.threads, maxtasksperchild=10)
    # smm = SharedMemoryManager()
    
    gizmos.validate_file_path(Options.feature_table, "feature table")
    gizmos.validate_file_path(Options.quantitation_matrix, "expression matrix")

    gizmos.print_milestone('Loading inputs...', Options.verbose)

    transcripts_df = pd.read_csv(Options.quantitation_matrix, index_col=0)
    metabolites_df = pd.read_csv(Options.feature_table, index_col=0)

    if Options.mad_filter:
        gizmos.print_milestone('Applying MAD filter to the counts...', Options.verbose)
        transcripts_df = apply_MAD_filter(transcripts_df)
        metabolites_df = apply_MAD_filter(metabolites_df)
    
    if Options.remove_zeros:
        gizmos.print_milestone('Removing arrays with at least one 0...', Options.verbose)
        transcripts_df = transcripts_df[(transcripts_df != 0).all(axis=1)]
        metabolites_df = metabolites_df[(metabolites_df != 0).all(axis=1)]
    
    gizmos.print_milestone('Checking column name integrity...', Options.verbose)
    transcripts_labels = set(transcripts_df.columns)
    metabolites_labels = set(metabolites_df.columns)
    common_labels = sorted(list(transcripts_labels.intersection(metabolites_labels)))

    #check common_labels variable
    if len(common_labels) == 0:
        raise ValueError("The labels in the transcriptome and metabolome files do not match. Please ensure the column names are consistent.")
    elif len(common_labels) <= 5:
        print(f"WARNING: Only {len(common_labels)} common labels found between the transcriptome and metabolome files. This may impact correlation analysis.")


    transcripts_df = transcripts_df[common_labels]
    metabolites_df = metabolites_df[common_labels]
    
    #Kumar 2nd Jan 2022
    #Manager class to make all process share the global variables.
    mgr = Manager()
    ns = mgr.Namespace()
    ns.df = transcripts_df
    results = []

    def append_result(result):
        results.append(result)
        pbar.update(1)

    total_iterations = len(metabolites_df)
    with tqdm(total=total_iterations, desc="Getting correlations") as pbar:
        for i, cur_metabolite in metabolites_df.iterrows():  
            pool.apply_async(corr_w_transcriptome, args=(cur_metabolite, ns.df, Options), error_callback=print_child_proc_error, callback=append_result)
        pool.close()
        pool.join()

    results_df = pd.concat(results).reset_index(drop=True)
    
    gizmos.print_milestone('Writing correlations to the database...', Options.verbose)
    
    gizmos.export_to_sql(Options.sqlite_db_name, results_df, Options.sqlite_table_name, index = False)

    if Options.mutual_rank and Options.method != "spearman":   
        gizmos.print_milestone('Calculating mutual rank statistic...', Options.verbose)
        
        #combine all metabolites and genes in a single non-redundant list
        metabolites = [i for i in list(results_df['metabolite'].squeeze()) if i is not None]
        genes = [i for i in list(results_df['gene'].squeeze()) if i is not None]
        all_objects = metabolites + genes
        all_objects = list(dict.fromkeys(all_objects))
        #
        edges_df = mutual_ranks.get_MR_from_correlations(results_df, all_objects)
        tablename = Options.sqlite_table_name + "_MR_edges"
        #edges_df.to_sql(tablename, conn, if_exists="append", index=False)
        gizmos.export_to_sql(Options.sqlite_db_name, edges_df, tablename, index = False)
        
        #        
        #Kumar
        #decay rate could be any value but generally it is either {5, 10, 25, 50, 100} from the Wisecaver (2017) paper.    
        #loop will generate 5 networks with 5 different decay rates
        if Options.multi_decay_rates:
            
            for d in Options.multi_decay_rates:
            
                weights_df = mutual_ranks.get_weight_from_MR(edges_df, Options.edge_weight_cutoff, d)
                
                tablename = Options.sqlite_table_name + "_MR_weights" + "_DR_{}".format(d)
                
                Options.weightsfile = os.path.join(script, 'sim_weights_DR_{}_file.csv'.format(d))
                
                print('Writing mutual ranks with decay rate of {} to the database...'.format(d))
                #weights_df.to_sql(tablename, conn, if_exists="append", index=False)
                gizmos.export_to_sql(Options.sqlite_db_name, weights_df, tablename, index = False)
                del weights_df['MR']
                
                #write the weights data to a file to be read later by ClusterOne
                weights_df.to_csv(Options.weightsfile, index=False, header=False, sep=' ')
            
                if Options.clusterone:
                    
                    Options.clusteronepath = os.path.join(script, 'cluster_one-1.0.jar')
                    Options.clusterone_outputfile = os.path.join(script, 'clusterOne_DR_{}.csv'.format(d))
                    
                    print('Generating modules with the decay rate of {} using ClusterOne...'.format(d))
                    mutual_ranks.run_clusterone(Options.clusterone_outputfile, Options.clusteronepath, Options.weightsfile)
                    
                    #move the clusterOne output file data to the sqlite db
                    clone_out_df = pd.read_csv(Options.clusterone_outputfile)
                    tablename = Options.sqlite_table_name + "_clone" + "_DR_{}".format(d)
                    #clone_out_df.to_sql(tablename, conn, if_exists="append", index=False)
                    gizmos.export_to_sql(Options.sqlite_db_name, clone_out_df, tablename, index = False)
                    
                    #remove clusterone and weights file from the current folder
                    os.remove(Options.clusterone_outputfile)
                    os.remove(Options.weightsfile)


        else:

            #Kumar
            #if multi_decay_rates is not set as True
            #The decay rate parameter could be used to generate edges with only one decay constant.
            #default is 25

            weights_df = mutual_ranks.get_weight_from_MR(edges_df, Options.edge_weight_cutoff, Options.decay_rate)
            tablename = Options.sqlite_table_name + "_MR_weights"
            Options.weightsfile = os.path.join(script, 'sim_weights_file.csv')
            gizmos.print_milestone('Writing mutual ranks to the database...', Options.verbose)
            #weights_df.to_sql(tablename, conn, if_exists="append", index=False)
            gizmos.export_to_sql(Options.sqlite_db_name, weights_df, tablename, index = False)
            del weights_df['MR']
            weights_df.to_csv(Options.weightsfile, index=False, header=False, sep=' ') 
        
            if Options.clusterone:
                Options.clusteronepath = os.path.join(script, 'cluster_one-1.0.jar')
                Options.clusterone_outputfile = os.path.join(script, 'clusterOne_DR_{}.csv'.format(Options.decay_rate))
                gizmos.print_milestone('Generating modules by clustering genes and metabolites using ClusterOne...', Options.verbose)
                mutual_ranks.run_clusterone(Options.clusterone_outputfile, Options.clusteronepath, Options.weightsfile)
                clone_out_df = pd.read_csv(Options.clusterone_outputfile)
                tablename = Options.sqlite_table_name + "_clone"
                #clone_out_df.to_sql(tablename, conn, if_exists="append", index=False)
                gizmos.export_to_sql(Options.sqlite_db_name, clone_out_df, tablename, index = False)
                # remove the physical files (these are space-delimited files)
                os.remove(Options.clusterone_outputfile)
                os.remove(Options.weightsfile)



if __name__ == "__main__":
    
    Options = get_args()
    main(Options)

