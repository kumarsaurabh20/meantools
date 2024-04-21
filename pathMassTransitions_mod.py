#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import argparse
import os
import sys
import numpy as np
import pandas as pd
import sqlite3

import gizmos


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--merged_clusters_file', default=True, action='store', required=True, help='Two-column csv with header. One mass signature per line. Format: ms_id,mz,mm')
    parser.add_argument('-t','--mass_transitions_file', default=True, action='store', required=True, help='Output from treatMassTansitions.py')
    parser.add_argument('-g', '--ghosts', required=False, default=False, action='store_true', help='Flag. Ghosts mass signatures will be added.')
    parser.add_argument('-dn', '--sqlite_db_name', default=True, action='store', required=True, help='Provide a name for the database!')
    parser.add_argument('-tn', '--sqlite_table_name', default=True, action='store', required=True, help='Provide a name for the database table!')
    parser.add_argument('-ct', '--sqlite_corr_tablename', default=True, action='store', required=True, help='Provide a name of the correlation table in the database!')
    parser.add_argument('-mt', '--sqlite_metabolite_tablename', default=True, action='store', required=True, help='Provide a name of the metabolite annotation table in the database!')
    parser.add_argument('-p','--pfam_RR_annotation_file',default=True, action='store', required=True, help='Nine-column csv. reaction_id, uniprot_id, Pfams, KO, rhea_id_reaction, kegg_id_reaction, rhea_confirmation, kegg_confirmation, KO_prediction')
    parser.add_argument('-a','--gene_annotation_file',default=True, action='store', required=True, help='Two-column csv. Gene, pfam1;pfam2')
    parser.add_argument('-s','--chemical_search_space', default='strict', required=False, choices=['strict', 'medium', 'loose'], help='Default: strict.')
    parser.add_argument('-cc','--corr_cutoff', default=0.7, required=False, type=float, help='Minimum absolute correlation coefficient. Default: 0.7. Use 0 for no cutoff.')
    parser.add_argument('-cpc','--corr_p_cutoff', default=0.1, required=False, type=float, help='Maximum P value of correlation. Default: 0.1. Use 1 for no cutoff.')
    parser.add_argument('-o', '--output_file', required=False, default=False, action='store', help='Output results in a CSV file!')
    parser.add_argument('--verbose', '-v', default=False, action='store_true', required=False)
    return parser.parse_args()


def load_enzyme_input(Options, correlation_df):
    """
    loads gene annotations, pfam-RR relationship file, and correlation and merges them.
    :return:
    """
    pfam_dict_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pfams_dict.csv')  # Acc,Name,Desc

    # GENE ANNOTATIONS
    # gene, pfam1;pfam2

    if Options.gene_annotation_file:
        gizmos.print_milestone('Loading gene annotations...', Options.verbose)
        annotations_df = pd.read_csv(Options.gene_annotation_file, index_col=None)

        annotations_df = annotations_df.rename(columns={annotations_df.columns[0]: 'gene', annotations_df.columns[1]: 'enzyme_pfams'})
        enzyme_pfams_list = annotations_df.enzyme_pfams.apply(gizmos.pd_to_list, separator=';')
        
        # expand gene annotations so there's one pfam per line (but we keep the "pfam" annotation that have them all)
        lens = [len(item) for item in enzyme_pfams_list]
        new_df = pd.DataFrame({'gene': np.repeat(annotations_df.gene, lens), 'pfam_rule': np.concatenate(enzyme_pfams_list)})

        # Kumar 07/08/2022
        # This step generates a df with 3 columns
        # gene;enzyme_pfams;pfam_rule
        # It makes one gene - one pfam entry
        annotations_df = pd.merge(annotations_df, new_df, how='outer')

        del enzyme_pfams_list, new_df
    else:
        annotations_df = pd.DataFrame()

    # Kumar 07/08/2022
    # PFAM - RR
    # reaction_id, uniprot_id, Pfams, KO, rhea_id_reaction, kegg_id_reaction, rhea_confirmation, kegg_confirmation,
    #  KO_prediction
    if Options.pfam_RR_annotation_file:
        gizmos.print_milestone('Loading pfam-RR annotations...', Options.verbose)
        pfam_rules_df = pd.read_csv(Options.pfam_RR_annotation_file, index_col=None)
        pfam_rules_df = pfam_rules_df.rename(columns={pfam_rules_df.columns[1]: 'uniprot_id', pfam_rules_df.columns[2]: 'uniprot_enzyme_pfams_acc'})
        pfam_rules_df['reaction_id'] = pfam_rules_df.reaction_id.astype('str')

        # filter type of anotations (strict, medium, loose)
        if Options.chemical_search_space == 'strict':
            pfam_rules_df = pfam_rules_df[pfam_rules_df.experimentally_validated]
        elif Options.chemical_search_space == 'medium':
            pfam_rules_df = pfam_rules_df[pfam_rules_df.experimentally_validated |
                                          pfam_rules_df.Pfam_ec_prediction]
        else:  # loose
            pass  # they are all there

        # convert pfam_acc to pfam
        pfam_dict = pd.read_csv(pfam_dict_file, index_col=None)
        pfam_dict.index = pfam_dict.Acc.apply(lambda x: x.split('.')[0])  # Acc,Name,Desc

        uniprot_enzyme_pfams_acc_list = pfam_rules_df.uniprot_enzyme_pfams_acc.apply(gizmos.pd_to_list, separator=' ')

        pfam_rules_df['uniprot_enzyme_pfams_list'] = [[k for k in row if k in pfam_dict.index] for row in uniprot_enzyme_pfams_acc_list]
        pfam_rules_df['uniprot_enzyme_pfams'] = pfam_rules_df.uniprot_enzyme_pfams_list.apply(';'.join)

        # Expand df so there is only one pfam per row.
        lens = [len(item) for item in pfam_rules_df.uniprot_enzyme_pfams_list]
        pfams_flat = [n for sub in pfam_rules_df.uniprot_enzyme_pfams_list for n in sub]

        new_df = pd.DataFrame({'uniprot_id': np.repeat(pfam_rules_df.uniprot_id, lens), 'pfam_rule': pfams_flat}).drop_duplicates()

        pfam_rules_df = pd.merge(pfam_rules_df, new_df, how='outer')  # on uniprot_id.
        del pfams_flat, uniprot_enzyme_pfams_acc_list, new_df
        del pfam_rules_df['uniprot_enzyme_pfams_list'], pfam_rules_df['uniprot_enzyme_pfams_acc']
        # ++ pfam_rule, uniprot_enzyme_pfams >>> -- enzyme_pfams_acc
    else:
        pfam_rules_df = pd.DataFrame()


    # MERGE
    if Options.gene_annotation_file and Options.pfam_RR_annotation_file:
        gizmos.print_milestone('Integrating annotations, and RR...', Options.verbose)

        merged_df = pd.merge(annotations_df, pfam_rules_df, how='inner')    # on pfam_rule

        del merged_df['pfam_rule']
        # now each rule has suspect genes
    elif Options.pfam_RR_annotation_file:
        merged_df = pfam_rules_df
        del merged_df['pfam_rule']
    else:
        merged_df = pd.DataFrame()

    
    gizmos.print_milestone('Integrating correlations...', Options.verbose)
    merged_df = pd.merge(merged_df, correlation_df, how='inner', on='gene')    # on gene


    gizmos.print_milestone('Duplicate cleanup...', Options.verbose)
    merged_df = merged_df.drop_duplicates()         # annotations_df merge produces duplicates due to pfam_rule

    return merged_df


def get_ghosts(masses_df, unique_transitions):
    """
    Identifies ghosts by using each reaction left and right of each mass, and finding matches.
    :param masses_df:
    :param unique_transitions:
    :return:
    """
    # forward reactions
    mm_fwd = np.vstack(masses_df.mm_round.to_numpy()) + unique_transitions
    mm_fwd = np.clip(mm_fwd, 0, None)
    mm_fwd = np.round(mm_fwd, 2)
    mm_fwd_df = pd.DataFrame(mm_fwd, index=masses_df.index, columns=[str(n) for n in unique_transitions])

    mm_fwd_df = mm_fwd_df.reset_index(drop=False)
    mm_fwd_df = pd.melt(mm_fwd_df, id_vars='metabolite', var_name='mass_transition_round', value_name='ghost_name')

    mm_fwd_df['ghost_name'] = mm_fwd_df.ghost_name.apply(lambda x: 'MM' + str(x))
    mm_fwd_df.rename(columns={'metabolite': 'substrate', 'ghost_name': 'product'}, inplace=True)
    mm_fwd_df = mm_fwd_df.astype({'mass_transition_round': float})
    mm_fwd_df = mm_fwd_df[mm_fwd_df.mass_transition_round > 0]
    mm_fwd_df['mass_transition_round'] = mm_fwd_df['mass_transition_round'].round(2)

    # backward reactions
    mm_bwd = np.vstack(masses_df.mm_round.to_numpy()) - unique_transitions
    mm_bwd = np.clip(mm_bwd, 0, None)
    mm_bwd = np.round(mm_bwd, 2)
    mm_bwd_df = pd.DataFrame(mm_bwd, index=masses_df.index, columns=[str(n) for n in unique_transitions])
    mm_bwd_df = mm_bwd_df.reset_index(drop=False)
    mm_bwd_df = pd.melt(mm_bwd_df, id_vars='metabolite', var_name='mass_transition_round', value_name='ghost_name')
    mm_bwd_df['ghost_name'] = mm_bwd_df.ghost_name.apply(lambda x: 'MM' + str(x))
    mm_bwd_df.rename(columns={'metabolite': 'product', 'ghost_name': 'substrate'}, inplace=True)
    mm_bwd_df = mm_bwd_df.astype({'mass_transition_round': float})
    mm_bwd_df = mm_bwd_df[mm_bwd_df.mass_transition_round > 0]
    mm_bwd_df['mass_transition_round'] = mm_bwd_df['mass_transition_round'].round(2)

    # OLD
    # match ghost mass
    #mm_fwd_df = mm_fwd_df[mm_fwd_df['product'].isin(mm_bwd_df['substrate'])]
    #mm_bwd_df = mm_bwd_df[mm_bwd_df['substrate'].isin(mm_fwd_df['product'])]

    # match ghost mass
    mm_fwd = mm_fwd_df[mm_fwd_df['product'].isin(mm_bwd_df['substrate'])]
    mm_bwd = mm_bwd_df[mm_bwd_df['substrate'].isin(mm_fwd_df['product'])]

    # Kumar
    # Nov 2023
    # Sometimes susbstrate and products do not match with their masses, like above which are ghosts
    # added a conditional to concat the mm_fwd_df and mm_bwd_df anyway.
    if mm_bwd.empty and mm_fwd.empty:

        # merge
        ghosts_df = pd.concat([mm_fwd_df, mm_bwd_df], sort=True)
        ghosts_df = ghosts_df.astype({'mass_transition_round': 'float'})
    else:
        # merge
        ghosts_df = pd.concat([mm_fwd, mm_bwd], sort=True)
        ghosts_df = ghosts_df.astype({'mass_transition_round': 'float'})
    
    return ghosts_df[['substrate', 'product', 'mass_transition_round']]


def get_transitions(masses_df, unique_transitions):
    """
    finds transitions that are possible
    :param masses_df:
    :param unique_transitions:
    :return:
    """
    path_transitions = np.vstack(masses_df.mm.to_numpy()) + unique_transitions

    path_transitions = np.round(path_transitions, 2)

    # Masses for each meatbolite are stacked from top to bottom using vstack()
    # Unique transition are added to each stack (vertical to horizontal merger)
    # Round the values
    # Because the hight of stack is similar to the length if masses_df.index. Decorate the stack using indexes
    # Then individual columns would be the mass transition that is added or subtracted (direction of the reaction)
    # column names are assigned based on unique mass transitions
    path_transitions_df = pd.DataFrame(path_transitions, index=masses_df.index, columns=[str(n) for n in unique_transitions])

    path_transitions_df = path_transitions_df.reset_index(drop=False)
    path_transitions_df = pd.melt(path_transitions_df, id_vars='metabolite', var_name='mass_transition', value_name='mm_round')
    path_transitions_df.rename(columns={'metabolite': 'substrate'}, inplace=True)

    # IMPORTANT
    # keep only products that are substrate by merging with masses
    path_transitions_df = pd.merge(path_transitions_df, masses_df.reset_index(drop=False), how='inner')
    # ^^ on mm_round

    # Kumar
    # Nov 2023
    # While checking Falcarindiol pathway elements
    # Sometimes path_transitions_sub_pro_df comes empty because the transitions mm_rounds do not match with the mm_round from masses_df
    # To bypass the empty dataframe, this step merges two dataframes using metabolite and substrate columns instead of mm_round. 
    #if path_transitions_sub_pro_df.empty:
    #    path_transitions_sub_pro_df = pd.merge(left=path_transitions_df, right=masses_df.reset_index(drop=False), left_on="substrate", right_on="metabolite", how='inner')
    if path_transitions_df.empty:
        gizmos.print_milestone('Enabling ghost mode...', Options.verbose)
        gizmos.print_milestone('Could not find direct Substrate-Product relationships so identifying ghosts...', Options.verbose)
        path_transitions_sub_prod_df = get_ghosts(masses_df, unique_transitions)

        return path_transitions_sub_prod_df[['substrate', 'product', 'mass_transition_round']].drop_duplicates()
    else:
        path_transitions_df.rename(columns={'metabolite': 'product'}, inplace=True)
        path_transitions_df = path_transitions_df.astype({'mass_transition': 'float'})
        path_transitions_df['mass_transition_round'] = path_transitions_df.mass_transition.apply(lambda x: np.round(x, 2))
        return path_transitions_df[['substrate', 'product', 'mass_transition_round']].drop_duplicates()


def filter_path_with_corr(path_transitions_df, enzyme_df):
    # merge on reaction_id, gene, and ms_substrate/product
    subs_corr_df = pd.merge(path_transitions_df, enzyme_df.rename(columns={'metabolite': 'substrate',
                                                                           'correlation': 'correlation_substrate',
                                                                           'P': 'P_substrate'}), how='inner')
    prod_corr_df = pd.merge(path_transitions_df, enzyme_df.rename(columns={'metabolite': 'product',
                                                                           'correlation': 'correlation_product',
                                                                           'P': 'P_product'}), how='inner')
    # final merge with outer allows for unilateral coexpression
    return pd.merge(subs_corr_df, prod_corr_df, how='outer')


def filter_transitions_with_corr(transitions_df, correlation_df):
    enzyme_df = load_enzyme_input(Options, correlation_df)

    enzyme_df = enzyme_df[['reaction_id', 'metabolite', 'gene']].drop_duplicates()

    # merge on metabolite, reaction_id
    try:
        merged_df = pd.merge(enzyme_df, transitions_df)
    except:
        print("ValueError: Check the data type of merging column!")
        sys.exit()

    # remove unnecessary columns, and duplicate cleanup
    merged_df = merged_df.drop(columns=['gene', 'metabolite']).drop_duplicates()
    return merged_df


#################
# MAIN PIPELINE #
#################

def main():
    # INPUT
    gizmos.print_milestone('Loading data...', Options.verbose)

    # OLD
    # Kumar 07/08/2022
    # mass signatures file
    # Output from queryMassNPDB2.py. It is not a query mass signature file. It contain repetitive mass features with
    # multiple structure's mm associated.
    # Test Wisecaver data file has ~2.8 million redundant mass features.
    #masses_df = pd.read_csv(Options.mass_signatures_file, index_col=None, header=0)
    #masses_df = masses_df.rename(columns={masses_df.columns[0]: 'metabolite', masses_df.columns[1]: 'mz', masses_df.columns[2]: 'mm'}).set_index('metabolite')
    #if 'mm_round' not in masses_df:
    #    masses_df['mm_round'] = round(masses_df.mm, Options.decimals)

    # NEW
    # Kumar Nov 2023
    # Connecting it with merged_clusters output in the new pipeline
    # Instead of queryMASSLotus or an old queryMassNPDB2.py output, the script now connects 
    # with the correlation step and takes the merged cluster file as input. With this file 
    # it extract the Mass signatures and the correlation matrix from the SQLite database using
    # import_from_sqlite() from gizmos.py.
    meregd_clusters_df = pd.read_csv(Options.merged_clusters_file, index_col='id', delimiter=',', quotechar='"')
    temp = meregd_clusters_df.to_dict(orient='list')
    

    # The temp has a dictionary structure but the values are present as string and not as 
    # individual elements.
    meregd_clusters_dict = {}
    for key, values in temp.items():
        new_key = key[:-1]  # Remove the trailing 's' from the key
        new_values = [item.strip() for value in values for item in value.split(',')]  # Split and flatten the values
        meregd_clusters_dict[new_key] = new_values
    
    #
    corr_columns = ['metabolite', 'gene', 'correlation', 'P']
    correlation_df = gizmos.import_from_sql(Options.sqlite_db_name, Options.sqlite_corr_tablename, corr_columns, conditions = meregd_clusters_dict, structures = False, clone = False)

    #
    # Extract mass signature from the queryMASSlotus script in the desired format using the same function
    # First extract the metabolites name in a dictionary like format
    # metabolite_dict = {'metabolite': meregd_clusters_dict['metabolite']}
    mass_sig_columns = ['metabolite', 'mz', 'molecular_weight']
    masses_df = gizmos.import_from_sql(Options.sqlite_db_name, Options.sqlite_metabolite_tablename, mass_sig_columns, conditions = meregd_clusters_dict, structures = True)
    masses_df = masses_df.rename(columns={"molecular_weight":"mm"}).set_index('metabolite')


    if 'mm_round' not in masses_df:
        masses_df['mm_round'] = round(masses_df.mm, 2)

    # Kumar 07/08/2022
    # This file has transitions from RetroRules database
    transitions_df = pd.read_csv(Options.mass_transitions_file, index_col=None)
    transitions_df['mass_transition_round_abs'] = transitions_df.mass_transition_round.apply(abs)

    # Kumar 07/08/2022
    # reaction_id is used to merge with the same from enzyme table. The dtypes are different in both the df
    # Therefore changing the dtype here to 'str'
    transitions_df['reaction_id'] = transitions_df.reaction_id.astype('str')

    # MINOR TREATMENTS
    mask = transitions_df.mass_transition_round_abs > 0
    transitions_df = transitions_df[mask]
    del transitions_df['mass_transition_round_abs']

    # FILTER TRANSITIONS WITH CORRELATION
    # Kumar 02/08/2022
    # This step merges all the PFAM annotations and omics correlations and filters out unwanted tarnsitions.
    if Options.pfam_RR_annotation_file and Options.gene_annotation_file:
        transitions_df = filter_transitions_with_corr(transitions_df, correlation_df)

    gizmos.print_milestone('Identifying transitions...', Options.verbose)
    # todo use mass_transition or round?
    unique_transitions = transitions_df.mass_transition.unique()

    # Kumar 07/08/2022
    # This step uses mass signature and unique transition from the transition file to filter and format mass signatures
    # based on transitions. Formating requires melting of data in to a particular format.
    path_transitions_df = get_transitions(masses_df, unique_transitions)


    # GHOSTS
    if Options.ghosts:
        gizmos.print_milestone('Making ghosts...', Options.verbose)
        path_transitions_df = pd.concat([path_transitions_df, get_ghosts(masses_df, unique_transitions)], sort=True)
        path_transitions_df = path_transitions_df.drop_duplicates()

    gizmos.print_milestone('Solving transitions...', Options.verbose)

    # todo possible memory improvement:
    #  cycle through mass_transition_round on different threads and write down immediately?
    solutions_df = pd.merge(transitions_df, path_transitions_df, how='inner')  # on mass_transition_round

    # OUTPUT
    gizmos.print_milestone('Writing...', Options.verbose)
    cols = ['substrate', 'product', 'reaction_id', 'mass_transition_round', 'mass_transition', 'substrate_id', 'substrate_mnx_id', 'substrate_mm', 'product_id', 'product_mnx_id', 'product_mm']
    
    if Options.output_file:
        solutions_df[cols].to_csv(Options.output_file, index=False)
    else:
        gizmos.export_to_sql(Options.sqlite_db_name, solutions_df[cols], Options.sqlite_table_name, index=False)
    


    return


if __name__ == "__main__":
    Options = get_args()
    Options.adducts_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'docs', 'ESI-MS-adducts.csv')
    main()
