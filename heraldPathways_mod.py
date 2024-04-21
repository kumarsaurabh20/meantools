#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool
from rdkit.Chem import AllChem as Chem

import gizmos


# TODO:
#  keep old structures in memory to properly recognize structures from another root,
#  even from same root when a deep step produces an early molecule
#  structures with multiple roots needs to be addressed. root needs to be smarter.
#  important when inputting two structures you expect to match (linoleic acid and falca)
#  Use enzyme filter to build ghosted-map of map transitions
#  Network: one node per structure


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--merged_clusters_file', default=True, action='store', required=True, help='Three column csv files <id, genes, metabolites>')
    parser.add_argument('-r','--rules', action='store', help='Output from validateRulesWithOrigins.py')
    parser.add_argument('-m','--RR_reaction_map', action='store', help='Output from mapBaseRetroRules.py (base_files.csv).')
    parser.add_argument('-p', '--pfam_RR_annotation', default='', action='store', required=False, help='Nine-column csv. reaction_id, uniprot_id, Pfams, KO, rhea_id_reaction, kegg_id_reaction, rhea_confirmation, kegg_confirmation, KO_prediction')
    parser.add_argument('-a', '--gene_annotation', default='', required=False, action='store',  help='Two-column csv. Gene, pfam1;pfam2')
    parser.add_argument('-d', '--pfam_dict', default='', required=False, action='store',  help='Three-column csv. Acc,Name,Desc')
    parser.add_argument('-s', '--chemical_search_space', default='strict', required=False, choices=['strict', 'medium', 'loose'], help='Default: strict.')
    parser.add_argument('-i', '--iterations', default=5, type=int, required=False, help='Number of iterations. Default: 5')
    parser.add_argument('-q', '--only_query_small', default=False, action='store_true', required=False, help='Use if we should query only small_rules.')
    parser.add_argument('-t', '--max_mass_transition_diff', default=0.05, type=float, required=False, help='Tolerance for the difference in expected and observed mass_transitions. Default = 0.05')
    parser.add_argument('-u', '--use_substrate_mm', default=False, action='store_true', required=False, help='Flag. Otherwise, mm is recalculated.')
    parser.add_argument('-dn', '--sqlite_db_name', default=True, action='store', required=True, help='Provide a name for the database!')
    parser.add_argument('-tn', '--sqlite_table_name', default=True, action='store', required=True, help='Provide a name for the database table!')
    parser.add_argument('-ct', '--sqlite_corr_tablename', default=True, action='store', required=True, help='Provide a name of the correlation table in the database!')
    parser.add_argument('-mt', '--sqlite_metabolite_tablename', default=True, action='store', required=True, help='Provide a name of the metabolite annotation table in the database!')
    parser.add_argument('-tt', '--sqlite_transition_tablename', default=True, action='store', required=True, help='Provide a name of the transitions table in the database!')
    parser.add_argument('-o', '--output_folder', action='store')
    parser.add_argument('-n', '--threads', default=4, type=int, required=False)
    parser.add_argument('-v', '--verbose', default=False, action='store_true', required=False)
    parser.add_argument('-dv', '--dev', default=False, action='store_true', required=False, help='Developer mode.')
    return parser.parse_args()


def load_enzyme_input(Options, mg_dict):
    """
    loads gene annotations, pfam-RR relationship file, and correlation and merges them.
    
    Pseudopipeline
    ---------------
    1. gene annotation
    2. RR df + PFAM annotation
    3. Integrate gene annotation with RR + PFAM (coonect genes through PFAMs; so genes have general reactions attached)
    4. Integrate correlations with RR + PFAM + genes
    
    :mg_dict: merged clusters files was converted in a dictionary format and passed here to generate correlation data frame

    :return: merged data frame with information from RR + PFAM + transition + correlation
    """
    pfam_dict_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pfams_dict.csv')  # Acc,Name,Desc

    # GENE ANNOTATIONS
    # gene, pfam1;pfam2

    if Options.gene_annotation:
        gizmos.print_milestone('Loading gene annotations...', Options.verbose)
        annotations_df = pd.read_csv(Options.gene_annotation, index_col=None)
        annotations_df = annotations_df.rename(columns={annotations_df.columns[0]: 'gene', annotations_df.columns[1]: 'enzyme_pfams'})
        
        # OLD
        #enzyme_pfams_list = annotations_df.enzyme_pfams.apply(gizmos.pd_to_list, separator=';')
        # expand gene annotations so there's one pfam per line (but we keep the "pfam" annotation that have them all)
        #lens = [len(item) for item in enzyme_pfams_list]
        #new_df = pd.DataFrame({'gene': np.repeat(annotations_df.gene, lens), 'pfam_rule': np.concatenate(enzyme_pfams_list)})

        annotations_df['enzyme_pfams'] = annotations_df['enzyme_pfams'].str.split(',')
        new_df = annotations_df.explode('enzyme_pfams').reset_index(drop=True)

        # Kumar 07/08/2022
        # This step generates a df with 3 columns
        # gene;enzyme_pfams;pfam_rule
        # It makes one gene - one pfam entry
        
        # OLD
        #annotations_df = pd.merge(annotations_df, new_df, how='outer')
        annotations_df = pd.merge(annotations_df, new_df, how="outer", on="gene")
        annotations_df = annotations_df.rename(columns={annotations_df.columns[1]: 'enzyme_pfams', annotations_df.columns[2]: 'pfam_rule'})
        annotations_df['enzyme_pfams'] = annotations_df['enzyme_pfams'].apply(lambda x: ','.join(map(str, x)))

        # OLD
        #del enzyme_pfams_list, new_df
        del new_df
    else:
        annotations_df = pd.DataFrame()

    # Kumar 07/08/2022
    # PFAM - RR
    # reaction_id, uniprot_id, Pfams, KO, rhea_id_reaction, kegg_id_reaction, rhea_confirmation, kegg_confirmation,
    #  KO_prediction
    if Options.pfam_RR_annotation:
        gizmos.print_milestone('Loading pfam-RR annotations...', Options.verbose)
        pfam_rules_df = pd.read_csv(Options.pfam_RR_annotation, index_col=None)
        pfam_rules_df = pfam_rules_df.rename(columns={pfam_rules_df.columns[1]: 'uniprot_id', pfam_rules_df.columns[2]: 'uniprot_enzyme_pfams_acc'})
        pfam_rules_df['reaction_id'] = pfam_rules_df.reaction_id.astype('str')

        # filter type of anotations (strict, medium, loose)
        if Options.chemical_search_space == 'strict':
            pfam_rules_df = pfam_rules_df[pfam_rules_df.experimentally_validated]
        elif Options.chemical_search_space == 'medium':
            pfam_rules_df = pfam_rules_df[pfam_rules_df.experimentally_validated | pfam_rules_df.Pfam_ec_prediction]
        else:  # loose
            pass  # they are all there

        # convert pfam_acc to pfam
        gizmos.print_milestone('Loading PFAM dictionary...', Options.verbose)
        pfam_dict = pd.read_csv(Options.pfam_dict, index_col=None)
        pfam_dict.index = pfam_dict.Acc.apply(lambda x: x.split('.')[0])  # Acc,Name,Desc


        # Kumar
        # Check if the seprater is working in this case
        # uniprot_enzyme_pfams_acc_list = pfam_rules_df.uniprot_enzyme_pfams_acc.apply(gizmos.pd_to_list, separator=' ')
        uniprot_enzyme_pfams_acc_list = pfam_rules_df.uniprot_enzyme_pfams_acc.apply(gizmos.pd_to_list, separator=';')

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

    # CORRELATION
    gizmos.print_milestone('Generating correlation table...', Options.verbose)
    corr_columns = ['metabolite', 'gene', 'correlation', 'P']
    correlation_df = gizmos.import_from_sql(Options.sqlite_db_name, Options.sqlite_corr_tablename, corr_columns, conditions = mg_dict, structures = False, clone = False)

    # MERGE
    
    # OLD
    #if Options.gene_annotation and Options.pfam_RR_annotation:
    #    gizmos.print_milestone('Integrating gene annotations with reaction rules...', Options.verbose)

    #    merged_df = pd.merge(annotations_df, pfam_rules_df, how='inner')    # on pfam_rule

    #    del merged_df['pfam_rule']
        # now each rule has suspect genes
    #elif Options.pfam_RR_annotation:
    #    merged_df = pfam_rules_df
    #    del merged_df['pfam_rule']
    #else:
    #    merged_df = pd.DataFrame()

    correlation_df.to_csv("correlation_df.csv", index=False)

    gizmos.print_milestone('Integrating gene annotations with reaction rules...', Options.verbose)
    merged_df = pd.merge(annotations_df, pfam_rules_df, how='inner')    # on pfam_rule
    del merged_df['pfam_rule']

    merged_df.to_csv("gene_annotations_with_reaction_rules.csv", index=False)

    #
    gizmos.print_milestone('Integrating correlations with gene annotations and reaction rules...', Options.verbose)
    merged_df = pd.merge(merged_df, correlation_df, how='inner')    # on gene

    merged_df.to_csv("correlations_gene_annotations_with_reaction_rules.csv", index=False)

    #
    gizmos.print_milestone('Duplicate cleanup...', Options.verbose)
    merged_df = merged_df.drop_duplicates()         # annotations_df merge produces duplicates due to pfam_rule

    merged_df.to_csv("correlations_gene_annotations_with_reaction_rules_drop_duplicates.csv", index=False)

    return merged_df


def load_and_merge_rules_and_transitions(base_smarts_id):
    """
    Loads and merges rules and transitions. Also returns the base_map df.
    :return:
    """
    # TRANSITIONS
    # ms,substrate,product,reaction_id,mass_transition_round,mass_transition,substrate_id,substrate_mnx_id,
    # substrate_mm,product_id,product_mnx_id,product_mm
    gizmos.print_milestone('Loading transitions...', Options.verbose)
    
    transition_columns = ['substrate', 'product', 'reaction_id', 'mass_transition_round', 'mass_transition', 'substrate_id', 'substrate_mnx_id', 'substrate_mm', 'product_id', 'product_mnx_id', 'product_mm']
    transitions_df = gizmos.import_from_sql(Options.sqlite_db_name, Options.sqlite_transition_tablename, transition_columns, conditions = {}, structures = False, clone = False)

    new_data_types = {'reaction_id': str,'substrate_id': str,'product_id': str}
    transitions_df = transitions_df.astype(new_data_types)
    
    transitions_df['reaction_substrate'] = transitions_df.reaction_id + '_' + transitions_df.substrate_id
    if 'substrate' in transitions_df.columns and 'product' in transitions_df.columns:
        Options.use_metabolomics = True
        transitions_df = transitions_df.rename(columns={'substrate': 'ms_substrate', 'product': 'ms_product',
                                                        'substrate_mnx_id': 'substrate_mnx',
                                                        'substrate_mm': 'substrate_mnx_mm',
                                                        'product_mnx_id': 'product_mnx',
                                                        'product_mm': 'product_mnx_mm'})
    else:
        Options.use_metabolomics = False
        transitions_df = transitions_df.rename(columns={'substrate_mnx_id': 'substrate_mnx',
                                                        'substrate_mm': 'substrate_mnx_mm',
                                                        'product_mnx_id': 'product_mnx',
                                                        'product_mm': 'product_mnx_mm'})
    if 'mass_transition_round' in transitions_df.columns:
        transitions_df = transitions_df.rename(columns={'mass_transition_round': 'expected_mass_transition'})
        del transitions_df['mass_transition']
    else:
        transitions_df = transitions_df.rename(columns={'mass_transition': 'expected_mass_transition'})

    # RR db (validated rules)
    # reaction_id, substrate_id, diameter, direction, smarts_id, rxn_smarts, validated
    # ++ rule_substrate_str, rule_product_str, reaction_substrate
    gizmos.print_milestone('Loading rules...', Options.verbose)
    rules_df = pd.read_csv(Options.rules, index_col=None, dtype={'reaction_id': str, 'substrate_id': str, 'validated': bool})

    # Kumar
    # select only validated reactions (118913/138832)
    rules_df = rules_df[rules_df.validated]

    rules_df['rule_substrate_str'] = rules_df.rxn_smarts.apply(gizmos.get_subs_string)
    rules_df['rule_product_str'] = rules_df.rxn_smarts.apply(gizmos.get_prod_string)
    rules_df['reaction_substrate'] = rules_df.reaction_id + '_' + rules_df.substrate_id
    rules_df = rules_df.rename(columns={'rule_substrate_str': 'RR_substrate_smarts', 'rule_product_str': 'RR_product_smarts'})

    # MOL
    df_to_mol = rules_df[['smarts_id', 'rxn_smarts']].drop_duplicates().copy()
    df_to_mol['rxn'] = df_to_mol.rxn_smarts.apply(Chem.ReactionFromSmarts)
    rules_df = pd.merge(rules_df, df_to_mol)
    del df_to_mol


    # BASE RULES
    gizmos.print_milestone('Getting base rules...', Options.verbose)
    base_rules_df = rules_df[rules_df.smarts_id.isin(base_smarts_id)][['smarts_id', 'rxn_smarts']].copy()
    base_rules_df = base_rules_df.drop_duplicates()
    base_rules_df['rxn'] = base_rules_df.rxn_smarts.apply(Chem.ReactionFromSmarts)

    # MERGE
    gizmos.print_milestone('Integrating rules and transitions...', Options.verbose)
    rt_df = pd.merge(rules_df, transitions_df, how='inner')    # on reaction_id, substrate_id

    return rt_df, base_rules_df


def load_input(mg_dict):
    """
    loads map, rules, transitions, and enzyme and correlation info if provided.
    :return:
    """
    # MAP RULES
    # smarts_id, smarts_is_in, smarts_has, identity, representative_smarts, is_base
    gizmos.print_milestone('Loading map...', Options.verbose)
    map_df = pd.read_csv(Options.RR_reaction_map, index_col=None, dtype={'is_base': bool})
    
    # # # convert to set
    map_df['smarts_is_in'] = map_df.smarts_is_in.apply(gizmos.pd_to_set, separator=';')
    map_df['smarts_has'] = map_df.smarts_has.apply(gizmos.pd_to_set, separator=';')
    map_df['identity'] = map_df.identity.apply(gizmos.pd_to_set, separator=';')
    
    #??????
    base_smarts_id = map_df.smarts_id[map_df.is_base]

    rt_df, base_rules_df = load_and_merge_rules_and_transitions(base_smarts_id)

    rt_df.to_csv("transition_and_rules.csv", index=False)

    # Kumar
    # 15/09/2022
    if Options.pfam_RR_annotation and Options.gene_annotation:
        
        enzyme_df = load_enzyme_input(Options, mg_dict)

        #merge on reaction_id and ms_substrate/product
        subs_corr_df = pd.merge(rt_df, enzyme_df.rename(columns={'metabolite': 'ms_substrate',
                                                                 'correlation': 'correlation_substrate',
                                                                 'P': 'P_substrate'}), how='inner')
        prod_corr_df = pd.merge(rt_df, enzyme_df.rename(columns={'metabolite': 'ms_product',
                                                                 'correlation': 'correlation_product',
                                                                 'P': 'P_product'}), how='inner')

        
        # Here, results may include same rule-gene-coexp data, but through different RR_enzyme annotation
        rt_df = pd.merge(subs_corr_df, prod_corr_df, how='outer')  # outer allows for unilateral coexpression
        
        # Now we use the allowed transitions to clean the map by removing requirements that are not in the allowed list
        # for this, we remove smarts_id from smarts_has
        allowed_smarts_id = set(rt_df.smarts_id)
        map_df['smarts_has'] = map_df.smarts_has.apply(lambda x: x.intersection(allowed_smarts_id))

        rt_df.to_csv("transition_merged_correlation_PFAM_reaction.csv", index=False)

    return rt_df, map_df, base_rules_df


def write_iter_summary(iters_log, is_init=False):
    """
    writes output detailing a summary of each iteration.
    :param iters_log:
    :param is_init:
    :return:
    """
    if is_init:
        with open(Options.summary_file, 'w') as f:
            f.write('iteration,initial_structures,fresh_structures,new_structures,total_structures\n')
    else:
        with open(Options.summary_file, 'a') as f:
            line = ','.join([iters_log['iteration'],
                             iters_log['initial_structures'],
                             iters_log['fresh_structures'],
                             iters_log['new_structures'],
                             iters_log['total_structures']])
            f.write(line + '\n')


def get_output_cols():
    """
    gets correct columns for output.
    :return:
    """
    cols_metabolomicless = ['expected_mass_transition', 'predicted_mass_transition', 'mass_transition_difference',
                            'predicted_substrate_id', 'predicted_product_id', 'root',
                            'reaction_id', 'substrate_id', 'product_id', 'substrate_mnx', 'product_mnx',
                            'smarts_id', 'diameter',
                            'predicted_substrate_smiles', 'predicted_product_smiles',
                            'RR_substrate_smarts', 'RR_product_smarts']

    cols_steps = ['ms_substrate', 'ms_product',
                  'expected_mass_transition', 'predicted_mass_transition', 'mass_transition_difference',
                  'predicted_substrate_id', 'predicted_product_id', 'root',
                  'reaction_id', 'substrate_id', 'product_id', 'substrate_mnx', 'product_mnx', 'smarts_id', 'diameter',
                  'predicted_substrate_smiles', 'predicted_product_smiles',
                  'RR_substrate_smarts', 'RR_product_smarts']


    cols_integrated = ['ms_substrate', 'ms_product',
                       'expected_mass_transition', 'predicted_mass_transition', 'mass_transition_difference',
                       'reaction_id', 'substrate_id', 'product_id',
                       'substrate_mnx', 'product_mnx',
                       'root', 'predicted_substrate_id', 'predicted_product_id',
                       'predicted_substrate_smiles', 'predicted_product_smiles',
                       'smarts_id', 'diameter',
                       'RR_substrate_smarts', 'RR_product_smarts',
                       'uniprot_id', 'uniprot_enzyme_pfams', 'KO',
                       'rhea_id_reaction', 'kegg_id_reaction',
                       'rhea_confirmation', 'kegg_confirmation', 'KO_prediction',
                       'gene', 'enzyme_pfams', 'correlation_substrate',
                       'P_substrate', 'correlation_product', 'P_product']

    return cols_metabolomicless, cols_steps, cols_integrated


def output_structures(structures_df, in_loop=True):
    """
    Outputs new structures after each loop.
    :param structures_df:
    :param in_loop:
    :return:
    """
    # structure_id, structure_mm, SMILES, [InChI], [reacted]
    cols = ['ms_substrate', 'predicted_substrate_id', 'predicted_substrate_mm', 'predicted_substrate_smiles',
            'reacted', 'root']
    if in_loop:
        # OUTPUT REACTED STRUCTURES
        reacted_df = structures_df[structures_df.reacted].drop(columns='predicted_substrate_mol')
        if not reacted_df.empty:
            gizmos.print_milestone('Writing structures...', Options.verbose)
            if not os.path.exists(Options.structures_output):
                reacted_df[cols].to_csv(Options.structures_output, index=None)
            else:
                reacted_df[cols].to_csv(Options.structures_output, index=None, mode='a', header=False)
            return
        else:
            gizmos.print_milestone('\nWriting final structures...', Options.verbose)
            if not structures_df.empty:
                if not os.path.exists(Options.structures_output):
                    structures_df[cols].to_csv(Options.structures_output, index=None)
                else:
                    structures_df[cols].to_csv(Options.structures_output, index=None, mode='a', header=False)
    else:
        gizmos.print_milestone('\nWriting final structures...', Options.verbose)
        if not structures_df.empty:
            if not os.path.exists(Options.structures_output):
                structures_df[cols].to_csv(Options.structures_output, index=None)
            else:
                structures_df[cols].to_csv(Options.structures_output, index=None, mode='a', header=False)
    return


def output_unique_structures(structures_list):
    structures = pd.DataFrame(structures_list, columns=['predicted_substrate_id',
                                                        'predicted_substrate_smiles',
                                                        'predicted_substrate_mol'])
    structures.drop(columns=['predicted_substrate_mol']).to_csv(Options.unique_structures_output, index=False)
    return


def output_reactions(steps_df):
    """
    Outputs reactions after each loop.
    :param steps_df:
    :return:
    """

    # these are just column headers
    cols_metabolomicless, cols_steps, cols_integrated = get_output_cols()


    # ENZYME SUPPORT
    if Options.pfam_RR_annotation and Options.gene_annotation:
        # OUTPUT DATA WITH ENZYMES
        if not steps_df.empty:
            if not os.path.exists(Options.reactions_output):
                steps_df[cols_integrated].to_csv(Options.reactions_output, index=None)
            else:
                steps_df[cols_integrated].to_csv(Options.reactions_output, index=None, mode='a', header=False)
    elif Options.use_metabolomics:
        # OUTPUT DATA WITHOUT ENZYMES
        if not steps_df.empty:
            if not os.path.exists(Options.reactions_output):
                steps_df[cols_steps].to_csv(Options.reactions_output, index=None)
            else:
                steps_df[cols_steps].to_csv(Options.reactions_output, index=None, mode='a', header=False)
    else:
        if not steps_df.empty:
            if not os.path.exists(Options.reactions_output):
                steps_df[cols_metabolomicless].to_csv(Options.reactions_output, index=None)
            else:
                steps_df[cols_metabolomicless].to_csv(Options.reactions_output, index=None, mode='a', header=False)
    return


def get_dfs_for_metabolite(cur_structure_id, structures_df, rt_df, map_df, base_rules_df):
    """
    gets map specific to metabolite based on allowed transitions.
    :param cur_structure_id:
    :param structures_df:
    :param rt_df:
    :param base_rules_df:
    :param map_df:
    :return:
    """

    cur_structure_df = structures_df[structures_df.predicted_substrate_id == cur_structure_id]

    if Options.use_metabolomics:
        rxn_df = rt_df.rename(columns={'ms_substrate_x': 'ms_substrate'})
        rxn_df = pd.merge(rxn_df, cur_structure_df)  # on ms_substrate
    else:
        rxn_df = rt_df.copy()
        rxn_df['predicted_substrate_id'] = cur_structure_id
        rxn_df = pd.merge(rxn_df, cur_structure_df)     # on predicted_substrate_id

    keep = set(rxn_df.smarts_id).union(base_rules_df.smarts_id)

    metabolite_map_df = map_df[map_df.smarts_id.isin(keep)].copy()
    metabolite_map_df['smarts_has'] = metabolite_map_df.smarts_has.apply(lambda x: x.intersection(keep))
    metabolite_map_df['identity'] = metabolite_map_df.identity.apply(lambda x: x.intersection(keep))

    new_base_mask = metabolite_map_df.smarts_has.apply(lambda x: len(x) == 0)
    new_base_smarts_id = metabolite_map_df.smarts_id[new_base_mask]
    metabolite_base_rules_df = rxn_df[rxn_df.smarts_id.isin(new_base_smarts_id)].copy()
    metabolite_base_rules_df = metabolite_base_rules_df[['smarts_id', 'rxn_smarts', 'rxn']].drop_duplicates()
    metabolite_base_rules_df = pd.concat([metabolite_base_rules_df, base_rules_df], sort=True)

    
    rxn_df.to_csv("rxn_df.csv", index=False)
    return rxn_df, metabolite_map_df, metabolite_base_rules_df


def check_direction(row):
    """
    Swaps id, smiles and mm of substrate and product when direction == -1.
    :param row:
    :return:
    """
    if row.direction == -1:
        orig_subs_id = row['predicted_substrate_id']
        orig_subs_smiles = row['predicted_substrate_smiles']
        orig_subs_mm = row['predicted_substrate_mm']

        orig_prod_id = row['predicted_product_id']
        orig_prod_smiles = row['predicted_product_smiles']
        orig_prod_mm = row['predicted_product_mm']

        row['predicted_substrate_id'] = orig_prod_id
        row['predicted_substrate_smiles'] = orig_prod_smiles
        row['predicted_substrate_mm'] = orig_prod_mm

        row['predicted_product_id'] = orig_subs_id
        row['predicted_product_smiles'] = orig_subs_smiles
        row['predicted_product_mm'] = orig_subs_mm
    return row


def load_structures(mg_dict):
    
    # STRUCTURES
    # [ms_name], structure_id, structure_mm, SMILES, [InChI], [reacted], [root]
    gizmos.print_milestone('Loading structures...', Options.verbose)

    # LOAD STRUCTURES
    # loads queryMassLotus ourput file
    structures_columns = ['metabolite', 'lotus_id', 'molecular_weight', 'smiles']
    structures_df = gizmos.import_from_sql(Options.sqlite_db_name, Options.sqlite_metabolite_tablename, structures_columns, conditions = mg_dict, structures = True)


    if len(structures_df.columns.intersection({'metabolite', 'ms_substrate'})):
        structures_df.rename(columns={structures_df.columns[0]: 'ms_substrate',
                                      structures_df.columns[1]: 'predicted_substrate_id',
                                      structures_df.columns[2]: 'predicted_substrate_mm',
                                      structures_df.columns[3]: 'predicted_substrate_smiles'}, inplace=True)

        cols = ['ms_substrate', 'predicted_substrate_id', 'predicted_substrate_mm', 'predicted_substrate_smiles']
        if 'reacted' in structures_df.columns:
            cols.append('reacted')
        if 'root' in structures_df.columns:
            cols.append('root')
        structures_df = structures_df[cols].drop_duplicates()
    else:
        Options.use_metabolomics = False
        structures_df.rename(columns={structures_df.columns[0]: 'predicted_substrate_id',
                                      structures_df.columns[1]: 'predicted_substrate_mm',
                                      structures_df.columns[2]: 'predicted_substrate_smiles'}, inplace=True)
        cols = ['predicted_substrate_id', 'predicted_substrate_mm', 'predicted_substrate_smiles']
        if 'reacted' in structures_df.columns:
            cols.append('reacted')
        if 'root' in structures_df.columns:
            cols.append('root')
        structures_df = structures_df[cols].drop_duplicates()

    # MOL'ING
    # moling on df with unique smiles so mols of same molecule get allocated same memory
    structures_to_mol = pd.DataFrame({'predicted_substrate_smiles':structures_df['predicted_substrate_smiles'].unique()})
    structures_to_mol['predicted_substrate_mol'] = structures_to_mol.predicted_substrate_smiles.apply(Chem.MolFromSmiles)

    structures_df = pd.merge(structures_df, structures_to_mol)

    del structures_to_mol

    # rewriting smiles with rdkit to ensure identification of identical structures through smiles
    structures_df['predicted_substrate_smiles'] = structures_df.predicted_substrate_mol.apply(Chem.MolToSmiles)
    
    # sometimes provided substrate_mm is inaccurate (or different from rdkits...)
    if not Options.use_substrate_mm:
        structures_df['predicted_substrate_mm'] = structures_df.predicted_substrate_mol.apply(gizmos.get_mm_from_mol, is_smarts=False)

    # Loop preparation ***** predicted_substrate_id is actually being added here as root
    if 'reacted' not in structures_df:
        structures_df['reacted'] = False
    if 'root' not in structures_df:
        structures_df['root'] = structures_df.predicted_substrate_id

    return structures_df


def update_structures(structures_df, structures_list):
    """
    Adds mols from the non-local variable.
    :param structures_df:
    :param structures_list:
    :return:
    """
    all_structures = pd.DataFrame(structures_list, columns=['predicted_substrate_id',
                                                            'predicted_substrate_smiles',
                                                            'predicted_substrate_mol'])

    structures_df = structures_df.drop(columns=['predicted_substrate_mol'])
    structures_df = pd.merge(structures_df, all_structures)
    return structures_df


def update_reactions_with_product_id(reactions_df, structures_list):
    # get structures smiles df

    structures = pd.DataFrame(structures_list, columns=['predicted_product_id','predicted_product_smiles','predicted_product_mol'])

    # Identify old structures through smiles
    fresh_structures_df = reactions_df.predicted_product_smiles.unique()
    fresh_structures_df = pd.DataFrame({'predicted_product_smiles': fresh_structures_df})
    fresh_structures_df = pd.merge(fresh_structures_df, structures, on="predicted_product_smiles", how='left')

    # ++ predicted_product_id, predicted_product_mol     => old structures identified. new have None

    # Name new structures
    new_structures_mask = fresh_structures_df.predicted_product_id.isna()
    n_to_name = len(fresh_structures_df[new_structures_mask])
    fresh_structures_df.loc[new_structures_mask, 'predicted_product_id'] = gizmos.generate_new_ids(
        n_to_name, all_ids=structures.predicted_product_id.unique())

    # Bring mols to fresh_structures_df without duplicates so same struct has same mol
    fresh_structures_mols = reactions_df[['predicted_product_smiles', 'predicted_product_mol']].drop_duplicates(
        subset=['predicted_product_smiles']).set_index('predicted_product_smiles')
    new_structures_smiles = fresh_structures_df.predicted_product_smiles[new_structures_mask]
    new_structure_mols = fresh_structures_mols.loc[new_structures_smiles].predicted_product_mol.tolist()
    fresh_structures_df.loc[new_structures_mask, 'predicted_product_mol'] = new_structure_mols
    # ++ predicted_product_mol  => mols added from reaction products on a same_memory manner.

    # update steps_df with new product_ids and same_memory mols
    updated_reactions_df = reactions_df.drop(columns=['predicted_product_mol'])
    updated_reactions_df = pd.merge(updated_reactions_df, fresh_structures_df)     # merge on product_smiles

    # get fully annotated new structures
    fresh_structures_df = get_new_structures(updated_reactions_df)

    updated_reactions_df.to_csv("updated_reactions_df.csv", index=False)
    fresh_structures_df.to_csv("fresh_structures_df.csv", index=False)

    return updated_reactions_df, fresh_structures_df


def get_new_structures(reactions_df):
    
    if Options.use_metabolomics:
        fresh_structures_df = reactions_df[['ms_product', 'predicted_product_id', 'predicted_product_smiles',
                                            'predicted_product_mm', 'predicted_product_mol', 'root']].drop_duplicates()
        fresh_structures_df = fresh_structures_df.rename(
            columns={'ms_product': 'ms_substrate',
                     'predicted_product_id': 'predicted_substrate_id',
                     'predicted_product_smiles': 'predicted_substrate_smiles',
                     'predicted_product_mm': 'predicted_substrate_mm',
                     'predicted_product_mol': 'predicted_substrate_mol'})
    else:
        fresh_structures_df = reactions_df[['predicted_product_id', 'predicted_product_smiles',
                                            'predicted_product_mm', 'predicted_product_mol', 'root']].drop_duplicates()
        fresh_structures_df = fresh_structures_df.rename(
            columns={'predicted_product_id': 'predicted_substrate_id',
                     'predicted_product_smiles': 'predicted_substrate_smiles',
                     'predicted_product_mm': 'predicted_substrate_mm',
                     'predicted_product_mol': 'predicted_substrate_mol'})
    return fresh_structures_df


def get_min_structure_data(structures_df):
    structures = structures_df[
        ['predicted_substrate_id', 'predicted_substrate_smiles', 'predicted_substrate_mol']].drop_duplicates()
    return structures.values.tolist()


def react_cur_structure(cur_structure_id, structures_df, rt_df, map_df, base_rules_df):
    rxn_df, cur_map_df, cur_base_rules_df = get_dfs_for_metabolite(cur_structure_id, structures_df, rt_df, map_df, base_rules_df)

    if rxn_df.empty:        # when the structure ID is not in the transitions
        return pd.DataFrame()
    else:
        process_results_df = gizmos.query_filtered_rxn_db(rxn_df, cur_map_df, cur_base_rules_df, Options)

        process_results_df.to_csv("process_results_df.csv", index=False)

        return process_results_df


def reaction_loop(rt_df, map_df, base_rules_df, mg_dict):
    """
    Main function that reacts structures and merges with enzyme data iteratively.
    :return:
    """
    def process_results(process_results_df, structures_list):
        if not process_results_df.empty:
            updated_process_results_df, fresh_structures_df = update_reactions_with_product_id(process_results_df, structures_list)

            # Update SMILES and mol (non-local)
            fresh_unique_structures_df = fresh_structures_df.drop_duplicates(subset=['predicted_substrate_id'])
            structures = pd.DataFrame(structures_list, columns=['predicted_substrate_id','predicted_substrate_smiles','predicted_substrate_mol'])
            new_structures_mask = fresh_unique_structures_df.predicted_substrate_id.isin(structures.predicted_substrate_id)
            new_structures_df = fresh_unique_structures_df[~new_structures_mask]
            for row in get_min_structure_data(new_structures_df):
                structures_list.append(row)

            # Update structures for next iteration (non-local)
            next_iter_structures.append(fresh_structures_df)

            # DIRECTION
            # Check reaction direction and swap substrate-products when -1
            updated_process_results_df = updated_process_results_df.apply(check_direction, axis=1)

            # OUTPUT
            output_reactions(updated_process_results_df)
        return

    # Kumar
    # QueryMASSLotus generates two types of file 1: metabolite, mz, molecular_weight 2. metabolite, lotus_id, molecular_weight, smiles
    # load_structure() takes file 2, rename the columns in to [ms_substrate, predicted_substrate_id, predicted_substrate_mm, predicted sustrate_smiles]
    # then generates mol from smiles using RDKit, also recalculate smiles (to validate) from generated mol using RDkit and lastly add two columns
    # "reacted" and "root". Root is similar to structure_id/predicted_substrate_id
    structures_df = load_structures(mg_dict)

    # next function gathers the followng three columns from the structures df and returns a list of values.
    # [predicted_substrate_id', 'predicted_substrate_smiles', 'predicted_substrate_mol'] i.e. substrate structure list with smiles
    structures_list = get_min_structure_data(structures_df)
    reacted_ms_structures = pd.DataFrame()

    i = 0
    while i < Options.iterations:
        # INIT
        i += 1
        gizmos.print_milestone('\nStarting iteration ' + str(i) + '...', Options.verbose)
        next_iter_structures = []

        # select structures to react
        # everything in recated column is FLASE, so essentially it takes all the rows.
        unreacted_mask = ~structures_df.reacted       # only unreacted structures
        structures_to_react = structures_df.predicted_substrate_id[unreacted_mask].unique()

        # Initial number of structures
        n_structures_initial = len(structures_to_react)

        # GENERATE VIRTUAL PRODUCTS
        gizmos.print_milestone(str(n_structures_initial) + ' structures will be reacted.', Options.verbose)
        gizmos.print_milestone('Generating products...', Options.verbose)
        
        if Options.dev:
            for cur_structure_id in structures_to_react:
                process_results(react_cur_structure(cur_structure_id, structures_df, rt_df, map_df, base_rules_df), structures_list)
        else: 

            # Kumar
            # Todo: multi processing is not working here
            # Check later without using --dev option
            with Pool(processes=Options.threads) as pool:
                for cur_structure_id in structures_to_react:
                    pool.apply_async(react_cur_structure, args=(cur_structure_id, structures_df, rt_df, map_df, base_rules_df), callback=process_results)
                pool.close()
                pool.join()

        # next_iter_structures is a list of dfs and has been updated by process_results
        if len(next_iter_structures):
            gizmos.print_milestone('Processing iteration results...', Options.verbose)

            # Update reacted ms_structures pairs
            structures_df['reacted'] = True
            reacted = structures_df[['ms_substrate', 'predicted_substrate_id', 'reacted']][structures_df.reacted]                
            reacted_ms_structures = pd.concat([reacted_ms_structures, reacted]).drop_duplicates()

            # Get structures of next iter and identify previously reacted ms_structure pairs
            next_iter_structures = pd.concat(next_iter_structures).drop_duplicates()
            next_iter_structures = pd.merge(next_iter_structures, reacted_ms_structures, how='outer') 
            
            # Adds False to the empty 'reacted' column
            next_iter_structures.loc[next_iter_structures.reacted.isna(), 'reacted'] = False

            # Output
            output_structures(structures_df)            # only reacted structures
            output_unique_structures(structures_list)   # all structures

            # Quick summary
            n_structures = len(next_iter_structures.predicted_substrate_id.unique())
            n_fresh_structures = n_structures - n_structures_initial

            # Update structures_df so next iteration can use it as input
            #structures_df = next_iter_structures[~next_iter_structures.reacted]
            structures_df = next_iter_structures[next_iter_structures.reacted == False]

            
            # Kumar
            # Condition to check if the structures with reacted attribute flase matches with True. 
            # If it matches then the next loop won't be executed else next iteration will begin. 
            for each in structures_df.predicted_substrate_id.unique():
                if each in next_iter_structures.predicted_substrate_id.unique():
                    gizmos.print_milestone('\nNo new structures were found, ending loop.', Options.verbose)
                    i = Options.iterations
                else:
                    pass

            n_new_structures = len(structures_df.predicted_substrate_id.unique())
            gizmos.print_milestone(str(n_fresh_structures) + ' structures generated.', Options.verbose)
            gizmos.print_milestone(str(n_new_structures) + ' new structures identified.', Options.verbose)
        else:   # here no new reactions were succesful.
            gizmos.print_milestone('\nNo new structures were found, ending loop.', Options.verbose)
            n_structures = n_structures_initial
            n_fresh_structures = 0
            n_new_structures = 0
            i = Options.iterations

        # ITERATION LOG
        iters_log = {'iteration': str(i),
                     'initial_structures': str(n_structures_initial),
                     'fresh_structures': str(n_fresh_structures),
                     'new_structures': str(n_new_structures),
                     'total_structures': str(n_structures)}
        write_iter_summary(iters_log)

    # FINAL OUTPUT
    output_structures(structures_df, in_loop=False)
    output_unique_structures(structures_list)  # all structures
    return



#################
# MAIN PIPELINE #
#################
def main():
    # global rt_df, map_df, base_rules_df


    meregd_clusters_df = pd.read_csv(Options.merged_clusters_file, index_col='id', delimiter=',', quotechar='"')
    temp = meregd_clusters_df.to_dict(orient='list')

    # The temp has a dictionary structure but the values are present as string and not as 
    # individual elements.
    mg_dict = {}
    for key, values in temp.items():
        new_key = key[:-1]  # Remove the trailing 's' from the key
        new_values = [item.strip() for value in values for item in value.split(',')]  # Split and flatten the values
        mg_dict[new_key] = new_values

    # OUTPUT INIT
    Options.log_file = os.path.join(Options.output_folder, 'log.txt')
    gizmos.log_init(Options)
    Options.summary_file = os.path.join(Options.output_folder, 'summary.csv')
    write_iter_summary({}, is_init=True)      # initializes file.

    # INPUT
    # TRANSITIONS
    # ms, ms_substrate, ms_product, reaction_id, mass_transition_round, mass_transition, substrate_id, substrate_mnx
    # substrate_mnx_mm, product_id, product_mnx, product_mnx_mm
    # STRUCTURES
    # ms_substrate, predicted_substrate_id, predicted_substrate_mm, predicted_substrate_smiles,
    # [InChI], [reacted], [root]
    # RULES
    # reaction_id, substrate_id, diameter, direction, smarts_id, rxn_smarts, validated
    # RR_substrate_smarts, RR_product_smarts, reaction_substrate
    # MAP
    # smarts_id, smarts_is_in, smarts_has, identity, representative_smarts, is_base
    rt_df, map_df, base_rules_df = load_input(mg_dict)

    # REACTION LOOP
    reaction_loop(rt_df, map_df, base_rules_df, mg_dict)

    # target:
    # [transitions]   ms_substrate, ms_product, expected_mass_transition_ms
    # [structure]     NPDB_id_substrate, NPDB_id_product
    # [reactions]     reaction_id, substrate_id, product_id, expected_mass_transition_rr, smarts_id_id, (direction)
    # [visualization] NPDB_substrate_smiles, NPDB_product_smiles, RR_substrate_smiles, RR_product_smiles

    return


if __name__ == "__main__":

    # global RT_DF, MAP_DF, BASE_RULES_DF

    Options = get_args()
    Options.use_metabolomics = False

    Options.unique_structures_output = os.path.join(Options.output_folder, 'structures.csv')
    Options.structures_output = os.path.join(Options.output_folder, 'structure_predictions.csv')
    Options.reactions_output = os.path.join(Options.output_folder, 'reactions.csv')

    Options.rejected_output_folder = os.path.join(Options.output_folder, 'rejected/')
    Options.rejected_structures_output = os.path.join(Options.rejected_output_folder, 'structures.csv')
    Options.rejected_reactions_output = os.path.join(Options.rejected_output_folder, 'reactions.csv')

    # RT_DF = pd.DataFrame()
    # MAP_DF = pd.DataFrame()
    # BASE_RULES_DF = pd.DataFrame()

    main()
