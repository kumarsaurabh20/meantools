#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from os.path import exists
import sqlite3
import numpy as np
import pandas as pd
from multiprocessing import Pool
from rdkit import Chem
from rdkit import rdBase
import seaborn as sns

import gizmos

rdBase.DisableLog('rdApp.*')

def get_args():
    """
    Self-explanatory. Toodles.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-rr_db', '--retrorules_sql_file', default=True, action='store', required=True, help='mvc.db')
    parser.add_argument('-mnx', '--metanetx_file', default=True, action='store', required=True, help='chem_prop.tsv')
    parser.add_argument('-m', '--monoisotopic_mass', default=False, action='store', required=False, help='Flag. Use if metanetx_file has a "mm" column.')
    parser.add_argument('-d', '--decimals', required=False, type=int, default=2, help='Number of decimals kept. Default: 2.')
    parser.add_argument('-o', '--output_folder', default=True, action='store', required=True)
    parser.add_argument('-ugr', '--use_greedy_rxn', default=False, action='store_true', required=False, help='Flag. Use to make reaction SMARTS greedier by removing H and valence requirements.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', required=False)
    parser.add_argument('-t', '--threads', action='store', default=4, type=int, required=False)
    return parser.parse_args()


def write_output(reaction_id, mass_transition_round, mass_transition, substrate_id, substrate_mnx_id, substrate_mm, product_id, product_mnx_id, product_mm, filename):
    line = ','.join([reaction_id, mass_transition_round, mass_transition, substrate_id, substrate_mnx_id, substrate_mm, product_id, product_mnx_id, product_mm])
    if not os.path.exists(Options.output_folder):
        os.makedirs(Options.output_folder)

    res_file = os.path.join(Options.output_folder, filename)

    with open(res_file, 'a') as f:
        f.write(line)
        f.write('\n')
    return


def add_rxn_smarts_and_validate(row, c):
    """
    Returns updated DFs that now contain smarts_id, diameter, smarts_string, greedy_rxn_smarts
    :param row:
    :param c:
    :return:
    """
    query = 'SELECT rules.smarts_id, rules.diameter, rules.direction, smarts.smarts_string FROM rules ' \
            'LEFT JOIN smarts ON rules.smarts_id = smarts.id ' \
            'WHERE isStereo=0 AND reaction_id=? AND substrate_id=?'
    to_query = (row.reaction_id, row.substrate_id)

    c.execute(query, to_query)
    rules = c.fetchall()
    rules = pd.DataFrame(rules, columns=['smarts_id', 'diameter', 'direction', 'smarts_string'],
                         index=np.repeat(row.name, len(rules)))     # row.name = index of parent df
    rules = pd.merge(row.to_frame().transpose(), rules, left_index=True, right_index=True)

    if Options.use_greedy_rxn:
        # Separate substrate and product from rxn SMARTS
        # We make greedy both substrate and product because RR misannotes them both.
        rules['rule_subs_smarts_str'] = rules.smarts_string.apply(gizmos.get_subs_string)
        rules['rule_prod_smarts_str'] = rules.smarts_string.apply(gizmos.get_prod_string)
        # Convert to mergeH smarts
        rules['rule_subs_smarts_mol'] = rules.rule_subs_smarts_str.apply(Chem.MolFromSmarts, mergeHs=True)
        rules['rule_prod_smarts_mol'] = rules.rule_prod_smarts_str.apply(Chem.MolFromSmarts, mergeHs=True)
        # Convert back to smarts
        rules['rule_subs_smarts_str'] = rules.rule_subs_smarts_mol.apply(Chem.MolToSmarts)
        rules['rule_prod_smarts_str'] = rules.rule_prod_smarts_mol.apply(Chem.MolToSmarts)
        # make SMARTS greedy, aka: remove valence and mergeh
        rules['rule_subs_smarts_str'] = rules.rule_subs_smarts_str.apply(gizmos.remove_valence_and_mergeh_from_smarts)
        rules['rule_prod_smarts_str'] = rules.rule_prod_smarts_str.apply(gizmos.remove_valence_and_mergeh_from_smarts)
        # Get mm
        rules['rxn_substrate_mm'] = rules.rule_subs_smarts_str.apply(gizmos.get_mm_from_str, is_smarts=True)
        # Join substrate and product into rxn SMARTS
        rules['greedy_rxn_smarts'] = rules[['rule_subs_smarts_str', 'rule_prod_smarts_str']].apply('>>'.join, axis=1)
        del rules['rule_subs_smarts_mol'], rules['rule_prod_smarts_mol']    # Memory management
        rules = rules.apply(gizmos.generate_virtual_molecule, axis=1,
                            rxn_smarts_name='greedy_rxn_smarts', substrate_smiles_name='SMILES')
    else:
        rules['rule_subs_smarts_str'] = rules.smarts_string.apply(gizmos.get_subs_string)
        rules['rxn_substrate_mm'] = rules.rule_subs_smarts_str.apply(gizmos.get_mm_from_str, is_smarts=True)
        rules = rules.apply(gizmos.generate_virtual_molecule, axis=1,
                            rxn_smarts_name='smarts_string', substrate_smiles_name='SMILES')
    rules['validated'] = False                              # Initialize
    mask = rules.predicted_product_mol_list.apply(bool)     # If there's a product, rxn SMARTS is valid
    rules.loc[mask, 'validated'] = True
    return rules


def query_bigger_smarts(cur_smarts, bigger_df):
    """
    For each cur_smarts it will identify which smarts are bigger, and will query them to see if cur_smarts is in each.
    Results are returned on a ;-separated string, or as "none" (NA).
    :param cur_smarts:
    :param bigger_df:
    :return:
    """
    this_smarts_is_in = bigger_df.mol.apply(lambda x: x.HasSubstructMatch(cur_smarts.mol))  # This is a bool Series
    this_smarts_is_in = bigger_df.smarts_id[this_smarts_is_in]

    if len(this_smarts_is_in):
        this_smarts_is_in = ';'.join([str(n) for n in this_smarts_is_in])
    else:
        this_smarts_is_in = ''
    results = ','.join([cur_smarts.smarts_id, this_smarts_is_in])
    return results


def get_representative_smarts(cur_rule, has_identical_smarts):
    """
    Groups cur_smarts and its identities to elect a representative smarts (smallest ID)
    :param cur_rule:
    :param has_identical_smarts:
    :return:
    """
    if cur_rule.smarts_id in has_identical_smarts:
        smarts_group = list(has_identical_smarts[cur_rule.smarts_id])
        smarts_group.append(cur_rule.smarts_id)
        smarts_group = sorted(smarts_group)
        return smarts_group[0]
    else:
        return cur_rule.smarts_id


def find_base_rules(cur_rule, has_smaller_smarts, has_identical_smarts):
    # ## Case 1 = yes has_smaller &             yes identical   =v
    # ## ## Case 1.1 = smaller == identity => choose representative base rule (smallest smarts_id)
    # ## ## Case 1.2 = smaller ~= identity => NOT base rule
    # ## Case 2 = yes has_smaller &             no  identical   => NOT base rule
    # ## Case 3 = no  has_smaller & yes is_in & yes identical   => choose representative base rule (smallest smarts_id)
    # ## Case 4 = no  has_smaller & yes is_in & no  identical   => base rule
    # ## Case 5 = no  has_smaller & no  is_in & yes identical   => choose representative base rule (smallest smarts_id)
    # ## Case 6 = no  has_smaller & no  is_in & no  identical   => base rule
    cur_rule['representative_smarts'] = get_representative_smarts(cur_rule, has_identical_smarts)

    if cur_rule.smarts_id in has_smaller_smarts:
        cur_rule['smarts_has'] = has_smaller_smarts[cur_rule.smarts_id]
        if cur_rule.smarts_id in has_identical_smarts:          # Case 1
            cur_rule['identity'] = has_identical_smarts[cur_rule.smarts_id]
            if has_smaller_smarts[cur_rule.smarts_id] == \
                    has_identical_smarts[cur_rule.smarts_id]:  # Case 1.1
                if cur_rule.smarts_id == cur_rule['representative_smarts']:
                    cur_rule['is_base'] = True
                else:
                    cur_rule['is_base'] = False
            else:                                                                                   # Case 1.2
                cur_rule['is_base'] = False
        else:                                                   # Case 2
            cur_rule['identity'] = None
            cur_rule['is_base'] = False
    elif cur_rule.smarts_id in has_identical_smarts:            # Case 3 & 5
        cur_rule['smarts_has'] = None
        cur_rule['identity'] = has_identical_smarts[cur_rule.smarts_id]
        if cur_rule.smarts_id == cur_rule['representative_smarts']:
            cur_rule['is_base'] = True
        else:
            cur_rule['is_base'] = False
    else:                                                       # Case 4 & 6
        cur_rule['smarts_has'] = None
        cur_rule['identity'] = None
        cur_rule['is_base'] = True

    # clean sets so only identity has identity and then convert to strings
    cur_rule['smarts_is_in'] = gizmos.pd_to_set(cur_rule.smarts_is_in, ';')
    if not pd.isnull(cur_rule.identity):
        if not pd.isnull(cur_rule.smarts_is_in):
            cur_rule['smarts_is_in'] = cur_rule.smarts_is_in - cur_rule.identity
        if not pd.isnull(cur_rule.smarts_has):
            cur_rule['smarts_has'] = cur_rule.smarts_has - cur_rule.identity

    # convert to strings
    cur_rule['identity'] = gizmos.set_to_string(cur_rule.identity, ';')
    cur_rule['smarts_has'] = gizmos.set_to_string(cur_rule.smarts_has, ';')
    cur_rule['smarts_is_in'] = gizmos.set_to_string(cur_rule.smarts_is_in, ';')

    return cur_rule


def write_output_results(results_row):
    """
    Writes output.
    :param results_row:
    :return:
    """
    res_file = os.path.join(Options.output_folder, 'all_small_rules.csv')
    with open(res_file, 'a') as resfile:
        resfile.write(results_row)
        resfile.write('\n')

    if not results_row.endswith(','):       # If there is a smarts_is_in
        network_file = os.path.join(Options.output_folder, 'network.csv')
        sub_smarts = results_row.split(',')[0]
        with open(network_file, 'a') as networkfile:
            for super_smarts in results_row.split(',')[-1].split(';'):
                line = ','.join([sub_smarts, super_smarts])
                networkfile.write(line)
                networkfile.write('\n')
    return


##################################
# Main database formatting steps #
##################################
def main():

    # OUTPUT FOLDER CREATION & LOG
    gizmos.log_init(Options)

    gizmos.print_milestone('Step 1: Mass Transitions', Options.verbose)
    gizmos.print_milestone('    Loading MetaNetX database...', Options.verbose)

    # Kumar
    # 4th January 2022
    # Headers have changed in the latest chem_prop.tsv compared to the old ones
    # so changed the mn_names according to the latest file
    if Options.monoisotopic_mass:
        mnx_names = ['MNX_ID', 'Name', 'Reference', 'Formula', 'Charge', 'Mass', 'InChI', 'InChIKey', 'SMILES', 'mm']
    else:
        mnx_names = ['MNX_ID', 'Name', 'Reference', 'Formula', 'Charge', 'Mass', 'InChI', 'InChIKey', 'SMILES']

    mnx_df = pd.read_csv(Options.metanetx_file, delimiter='\t', comment='#', header=None, names=mnx_names, index_col=0)
    if 'mm' not in mnx_df.columns:
        gizmos.print_milestone('    Calculating monoisotopic mass...', Options.verbose)
        mnx_df['mm'] = mnx_df.SMILES.apply(gizmos.get_mm_from_str, is_smarts=False, rounded=False)
        fname = Options.metanetx_file[:-4] + '.mm_treated.tsv'
        if not os.path.exists(Options.output_folder):
            os.makedirs(Options.output_folder)
        temp_file_path = os.path.join(Options.output_folder, fname)
        mnx_df.to_csv(temp_file_path, index=True, sep='\t')

    gizmos.print_milestone('    Querying RetroRules database...', Options.verbose)
    conn = sqlite3.connect(Options.retrorules_sql_file)
    c1 = conn.cursor()
    c2 = conn.cursor()

    results = {}  # {reaction_id:{s:set(),p:set()} }

    # reaction_substrates
    query = 'SELECT * FROM reaction_substrates WHERE is_main=1'
    for reaction_id, chemical_id, stochiometry, is_main in c1.execute(query):
        if reaction_id not in results:
            results[reaction_id] = {'substrate_id': set(), 'product_id': set()}
        results[reaction_id]['substrate_id'].add(chemical_id)

    # reaction_products
    # reaction_id, chemical_id, stochiometry, is_main
    query = 'SELECT * FROM reaction_products WHERE is_main=1'
    for reaction_id, chemical_id, stochiometry, is_main in c1.execute(query):
        if reaction_id not in results:
            results[reaction_id] = {'substrate_id': set(), 'product_id': set()}
        results[reaction_id]['product_id'].add(chemical_id)

    # OUTPUT INIT
    header = ','.join(
        ['reaction_id', 'mass_transition_round', 'mass_transition', 'substrate_id', 'substrate_mnx_id', 'substrate_mm',
         'product_id', 'product_mnx_id', 'product_mm'])
    file_path = os.path.join(Options.output_folder, "MassTransitions.csv")
    with open(file_path, 'w') as f:
        f.write(header)
        f.write('\n')

    gizmos.print_milestone('    Identifying transitions...', Options.verbose)
    # rule_products
    # product_id, reaction_id, substrate_id, diameter, isStereo, stochiometry
    query1 = 'SELECT substrate_id, product_id FROM rule_products WHERE diameter=16 and isStereo=0 and reaction_id=?'
    query2 = 'SELECT mnxm FROM chemical_species WHERE id=?'
    temp = []
    for cur_reaction_id in results:
        for substrate_id, product_id in c1.execute(query1, (str(cur_reaction_id),)):
            if (substrate_id in results[cur_reaction_id]['substrate_id'] and
                    product_id in results[cur_reaction_id]['product_id']):
                # substrate
                c2.execute(query2, (str(substrate_id),))
                substrate_mnx_id = c2.fetchone()[0]
                if substrate_mnx_id not in mnx_df.index:
                    continue
                substrate_mm = mnx_df.mm[substrate_mnx_id]
                # product
                c2.execute(query2, (str(product_id),))
                product_mnx_id = c2.fetchone()[0]
                if product_mnx_id not in mnx_df.index:
                    continue
                product_mm = mnx_df.mm[product_mnx_id]
                if np.isnan(substrate_mm) or np.isnan(product_mm):
                    continue
                mass_transition = product_mm - substrate_mm
                mass_transition_round = round(mass_transition, Options.decimals)
                write_output(str(cur_reaction_id), str(mass_transition_round), str(mass_transition),
                             str(substrate_id), substrate_mnx_id, str(substrate_mm),
                             str(product_id), product_mnx_id, str(product_mm), "MassTransitions.csv")

###########################
### TreatMassTransition ###
###########################
    gizmos.print_milestone('    Writing other files...', Options.verbose)
    file_path = os.path.join(Options.output_folder, "MassTransitions.csv")
    if exists(file_path):
        df = pd.read_csv(file_path, index_col=None)
        df['mass_transition_round'] = df.mass_transition.apply(round, args=(Options.decimals,))
        df = df[
            ['reaction_id', 'mass_transition_round', 'mass_transition', 'substrate_id', 'substrate_mnx_id', 'substrate_mm',
            'product_id', 'product_mnx_id', 'product_mm']]
        df = df.dropna(how='any')

        df2 = dict()
        for i, row in df.iterrows():
            if row.mass_transition_round not in df2:
                df2[row.mass_transition_round] = 0
            df2[row.mass_transition_round] += 1

        df2 = pd.DataFrame.from_dict(df2, orient='index')
        df2.reset_index(drop=False, inplace=True)
        df2.columns = ['mass_transition', 'n_reactions']

    # OUTPUT

        fname = os.path.join(Options.output_folder, 'mass_transitions.rounded.csv')
        df.to_csv(fname, index=False)

        fname = os.path.join(Options.output_folder, 'mass_transitions_counts.csv')
        df2.to_csv(fname, index=False)

    # PLOT
        sns_plot = sns.scatterplot(data=df2, x='mass_transition', y='n_reactions')
        fig = sns_plot.get_figure()
        fname = os.path.join(Options.output_folder, 'transitions_degeneracy.png')
        fig.savefig(fname)
    else:
        print("MassTransitions file does not exist!")
        exit()

##################################
### Validate Rules with Origin ###
##################################
    gizmos.print_milestone('Step 2: Validate Rules', Options.verbose)
    gizmos.print_milestone('    Loading input...', Options.verbose)
    mnx_names = ['MNX_ID', 'Name', 'Reference', 'Formula', 'Charge', 'Mass', 'InChI', 'InChIKey', 'SMILES']
    mnx_df = pd.read_csv(Options.metanetx_file, delimiter='\t', comment='#', header=None, names=mnx_names, index_col=None)
    # MNX_ID	Description	Formula	Charge	Mass	InChI	SMILES	Source	InChIKey

    file_path = os.path.join(Options.output_folder, "MassTransitions.csv")
    transitions_df = pd.read_csv(file_path, index_col=None)
    # reaction_id,mass_transition_round,mass_transition,substrate_id,substrate_mnx_id,substrate_wt,product_id,product_mnx_id,product_wt

    conn = sqlite3.connect(Options.retrorules_sql_file)
    c = conn.cursor()

    # Convert transitions_df into only 5 or 6 columns (stack id,mnx,wt data)
    try:  # in case we're using treated transitions
        # substrate_id,substrate_mnx_id,substrate_mm,product_id,product_mnx_id,product_mm
        # tmp_df = transitions_df[['reaction_id', 'mass_transition_round', 'mass_transition', 'product_id', 'product_mnx_id', 'product_wt']].copy()

        # Kumar
        # 5th January 2022
        # Product_wt field name is not present in the MassTransition file
        tmp_df = transitions_df[
            ['reaction_id', 'mass_transition_round', 'mass_transition', 'product_id', 'product_mnx_id']].copy()
    except KeyError:  # in case we're using original transitions
        # tmp_df = transitions_df[['reaction_id', 'mass_transition', 'product_id', 'product_mnx_id', 'product_wt']].copy()
        tmp_df = transitions_df[['reaction_id', 'mass_transition', 'product_id', 'product_mnx_id']].copy()
    # tmp_df = tmp_df.rename(columns={'product_id': 'substrate_id', 'product_mnx_id': 'substrate_mnx_id', 'product_wt': 'substrate_wt'})
    tmp_df = tmp_df.rename(columns={'product_id': 'substrate_id', 'product_mnx_id': 'substrate_mnx_id'})

    # transitions_df = transitions_df.drop(columns=['product_id', 'product_mnx_id', 'product_wt'])
    transitions_df = transitions_df.drop(columns=['product_id', 'product_mnx_id'])
    transitions_df = pd.concat([transitions_df, tmp_df])

    del tmp_df

    # merge transitions with mnx data and mol smiles
    gizmos.print_milestone('    Merging...', Options.verbose)
    df = pd.merge(transitions_df, mnx_df, how='left', left_on='substrate_mnx_id', right_on='MNX_ID')
    df = df.dropna(how='any', subset=['SMILES']).reset_index(drop=True)
    del transitions_df, mnx_df

    gizmos.print_milestone("    Mol'ing structures...", Options.verbose)
    df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
    df = df.dropna(how='any', subset=['mol']).reset_index(drop=True)

    gizmos.print_milestone('    Validating...', Options.verbose)
    # Loop through each reaction_substrate, check against RR. Only substrate (which includes products as -1)
    df = df.apply(add_rxn_smarts_and_validate, args=(c,), axis=1)  # This is a series of DFs
    df = pd.concat(df.tolist()).reset_index(drop=True)  # This is a DF.

    # clean columns
    gizmos.print_milestone('    Output...', Options.verbose)
    cols = ['reaction_id', 'substrate_id', 'diameter', 'direction',
            'smarts_id', 'rxn_smarts', 'rxn_substrate_mm', 'validated']
    if Options.use_greedy_rxn:
        df = df.rename(columns={'greedy_rxn_smarts': 'rxn_smarts'})
    else:
        df = df.rename(columns={'smarts_string': 'rxn_smarts'})
    val_path = os.path.join(Options.output_folder, "ValidateRulesWithOrigins.csv")
    df[cols].to_csv(val_path, index=False)
    # reaction_id,substrate_id,diameter,direction,smarts_id,rxn_smarts,validated

###########################
### Map Base RetroRules ###
###########################

    # LOAD INPUT
    gizmos.print_milestone('Step 3: MapBase RetroRules', Options.verbose)
    gizmos.print_milestone('    Loading input...', Options.verbose)
    df = pd.read_csv(val_path, index_col=None,
                     dtype={'reaction_id': str, 'substrate_id': str, 'validated': bool})
    # ^^ reaction_id,substrate_id,diameter,direction,smarts_id,rxn_smarts,validated
    df = df[df.validated].reset_index(drop=True)  # Ignore unvalidated rules
    df['reaction_substrate'] = df.reaction_id + '_' + df.substrate_id

    # DEFINING SMALL RULES
    gizmos.print_milestone('    Defining small rules...', Options.verbose)
    all_small_rules = []
    for cur_reaction_substrate in df.reaction_substrate.unique():  # Loop through reaction_substrate
        mask = df.reaction_substrate == cur_reaction_substrate
        index_of_smallest_diameter = df[mask].sort_values(by='diameter').iloc[0].name  # sort ascending by diameter
        all_small_rules.append(index_of_smallest_diameter)  # ..and pick first row.

    all_small_rules = df.loc[all_small_rules].copy().reset_index(drop=True)  # This is a df of small_rules only
    del df

    gizmos.print_milestone('    ' + str(len(all_small_rules)) + ' structures to query found...', Options.verbose)

    # PRE-PROCESSING
    gizmos.print_milestone("    Mol'ing...", Options.verbose)
    all_small_rules['substrate_str'] = all_small_rules.rxn_smarts.apply(gizmos.get_subs_string)
    # Create a df specifically for mols so we don't mol same structure twice. Index=smarts_id
    mols_df = all_small_rules.drop_duplicates(subset=['smarts_id'], keep='first')[['smarts_id', 'substrate_str']]
    mols_df = mols_df.set_index('smarts_id')
    mols_df['mol'] = mols_df.substrate_str.apply(Chem.MolFromSmarts)
    mols_df['mm'] = mols_df.mol.apply(gizmos.get_mm_from_mol, is_smarts=True)  # Mass of the pattern.
    # Merge mols_df into original df to continue normal pipeline.
    del mols_df['substrate_str']  # to avoid duplicate column
    all_small_rules = pd.merge(all_small_rules, mols_df, left_on='smarts_id', right_index=True)
    del mols_df

    # INITIALIZING OUTPUT
    res_file = os.path.join(Options.output_folder, 'all_small_rules.csv')
    with open(res_file, 'w') as resfile:
        resfile.write('smarts_id,smarts_is_in')
        resfile.write('\n')
    network_file = os.path.join(Options.output_folder, 'network.csv')
    with open(network_file, 'w') as networkfile:
        networkfile.write('subs,super')
        networkfile.write('\n')

    # FINDING SMALL RULES RELATIONSHIPS (IS_IN)
    gizmos.print_milestone('    Querying...', Options.verbose)
    with Pool(processes=Options.threads) as pool:
        for i, cur_rule in all_small_rules.iterrows():
            mm_mask = all_small_rules.mm >= cur_rule.mm
            not_same_mask = all_small_rules.smarts_id != cur_rule.smarts_id
            bigger_df = all_small_rules[mm_mask & not_same_mask]
            pool.apply_async(query_bigger_smarts, args=(cur_rule, bigger_df), callback=write_output_results)
        pool.close()
        pool.join()
    del all_small_rules

    # DEFINING BASE RULES
    small_rules_df = pd.read_csv(res_file, index_col=None, dtype={'smarts_id': str, 'smarts_is_in': str})
    small_rules_df = small_rules_df.drop_duplicates(subset='smarts_id').set_index('smarts_id')
    small_rules_df.to_csv(res_file, index=True)

    # # find identities
    gizmos.print_milestone('    Processing identities...', Options.verbose)
    has_smaller_smarts = {}
    has_identical_smarts = {}
    for smarts_id, cur_rule in small_rules_df.iterrows():
        if pd.isnull(cur_rule.smarts_is_in):  # no matches = no identities
            continue
        else:
            for bigger_smarts in cur_rule.smarts_is_in.split(';'):  # loop through all bigger_smarts
                if bigger_smarts not in has_smaller_smarts:
                    has_smaller_smarts[bigger_smarts] = set()  # Init
                has_smaller_smarts[bigger_smarts].add(smarts_id)  # Here we fill smarts_has
                if (not pd.isnull(small_rules_df.smarts_is_in[bigger_smarts]) and  # bigger_smarts has bigger_smarts?
                        smarts_id in small_rules_df.smarts_is_in[bigger_smarts]):  # bigger_smarts is cur_smarts?
                    if bigger_smarts not in has_identical_smarts:
                        has_identical_smarts[bigger_smarts] = set()
                    if smarts_id not in has_identical_smarts:
                        has_identical_smarts[smarts_id] = set()
                    has_identical_smarts[bigger_smarts].add(smarts_id)  # then we add to identities
                    has_identical_smarts[smarts_id].add(bigger_smarts)

    # # find base rules
    # ## Case 1 = yes has_smaller &             yes identical   =v
    # ## ## Case 1.1 = smaller == identity => choose representative base rule (smallest smarts_id)
    # ## ## Case 1.2 = smaller ~= identity => NOT base rule
    # ## Case 2 = yes has_smaller &             no  identical   => NOT base rule
    # ## Case 3 = no  has_smaller & yes is_in & yes identical   => choose representative base rule (smallest smarts_id)
    # ## Case 4 = no  has_smaller & yes is_in & no  identical   => base rule
    # ## Case 5 = no  has_smaller & no  is_in & yes identical   => choose representative base rule (smallest smarts_id)
    # ## Case 6 = no  has_smaller & no  is_in & no  identical   => base rule
    gizmos.print_milestone('    Finding base rules...', Options.verbose)
    small_rules_df = small_rules_df.rename_axis('smarts_id').reset_index()
    small_rules_df = small_rules_df.apply(find_base_rules,
                                          args=(has_smaller_smarts, has_identical_smarts,), axis=1)

    small_rules_df = small_rules_df[['smarts_id', 'smarts_is_in', 'smarts_has', 'identity',
                                     'representative_smarts', 'is_base']]
    res_file = os.path.join(Options.output_folder, 'base_rules.csv')
    small_rules_df.to_csv(res_file, index=False)
    # ^^ smarts_id, smarts_is_in, smarts_has, identity, representative_smarts, is_base

    return


if __name__ == "__main__":
    Options = get_args()
    Options.log_file = os.path.join(Options.output_folder, 'log.txt')
    main()
