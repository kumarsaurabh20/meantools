#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import argparse
import sqlite3
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import rdBase

import gizmos

rdBase.DisableLog('rdApp.*')


def get_args():
    """
    Self-explanatory. Toodles.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('retrorules_sql_file')
    parser.add_argument('mass_transitions_file',
                        help='Output from getMassTransitions')
    parser.add_argument('metanetx_file',
                        help='chem_prop.tsv')
    parser.add_argument('output_file')
    parser.add_argument('--use_greedy_rxn',
                        default=False,
                        action='store_true',
                        required=False,
                        help='Flag. Use to make reaction SMARTS greedier by removing H and valence requirements.')
    parser.add_argument('--verbose', '-v',
                        default=False,
                        action='store_true',
                        required=False)
    return parser.parse_args()


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


#################
# MAIN PIPELINE #
#################
def main():
    # LOAD INPUT
    gizmos.print_milestone('Loading input...', Options.verbose)
    mnx_names = ['MNX_ID', 'Name', 'Reference', 'Formula', 'Charge', 'Mass', 'InChI', 'InChIKey', 'SMILES']
    mnx_df = pd.read_csv(Options.metanetx_file, delimiter='\t', comment='#', header=None, names=mnx_names,
                         index_col=None)
    # MNX_ID	Description	Formula	Charge	Mass	InChI	SMILES	Source	InChIKey

    transitions_df = pd.read_csv(Options.mass_transitions_file, index_col=None)
    # reaction_id,mass_transition_round,mass_transition,substrate_id,substrate_mnx_id,substrate_wt,product_id,product_mnx_id,product_wt

    conn = sqlite3.connect(Options.retrorules_sql_file)
    c = conn.cursor()

    # Convert transitions_df into only 5 or 6 columns (stack id,mnx,wt data)
    try:                # in case we're using treated transitions
        # substrate_id,substrate_mnx_id,substrate_mm,product_id,product_mnx_id,product_mm
        # tmp_df = transitions_df[['reaction_id', 'mass_transition_round', 'mass_transition', 'product_id', 'product_mnx_id', 'product_wt']].copy()

        # Kumar
        # 5th January 2022
        # Product_wt field name is not present in the MassTransition file
        tmp_df = transitions_df[['reaction_id', 'mass_transition_round', 'mass_transition', 'product_id', 'product_mnx_id']].copy()
    except KeyError:    # in case we're using original transitions
        # tmp_df = transitions_df[['reaction_id', 'mass_transition', 'product_id', 'product_mnx_id', 'product_wt']].copy()
        tmp_df = transitions_df[['reaction_id', 'mass_transition', 'product_id', 'product_mnx_id']].copy()
    # tmp_df = tmp_df.rename(columns={'product_id': 'substrate_id', 'product_mnx_id': 'substrate_mnx_id', 'product_wt': 'substrate_wt'})
    tmp_df = tmp_df.rename(columns={'product_id': 'substrate_id', 'product_mnx_id': 'substrate_mnx_id'})

    # transitions_df = transitions_df.drop(columns=['product_id', 'product_mnx_id', 'product_wt'])
    transitions_df = transitions_df.drop(columns=['product_id', 'product_mnx_id'])
    transitions_df = pd.concat([transitions_df, tmp_df])

    del tmp_df

    # merge transitions with mnx data and mol smiles
    gizmos.print_milestone('Merging...', Options.verbose)
    df = pd.merge(transitions_df, mnx_df, how='left', left_on='substrate_mnx_id', right_on='MNX_ID')
    df = df.dropna(how='any', subset=['SMILES']).reset_index(drop=True)
    del transitions_df, mnx_df

    gizmos.print_milestone("Mol'ing structures...", Options.verbose)
    df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
    df = df.dropna(how='any', subset=['mol']).reset_index(drop=True)

    gizmos.print_milestone('Validating...', Options.verbose)
    # Loop through each reaction_substrate, check against RR. Only substrate (which includes products as -1)
    df = df.apply(add_rxn_smarts_and_validate, args=(c,), axis=1)               # This is a series of DFs
    df = pd.concat(df.tolist()).reset_index(drop=True)                          # This is a DF.

    # clean columns
    gizmos.print_milestone('Output...', Options.verbose)
    cols = ['reaction_id', 'substrate_id', 'diameter', 'direction',
            'smarts_id', 'rxn_smarts', 'rxn_substrate_mm', 'validated']
    if Options.use_greedy_rxn:
        df = df.rename(columns={'greedy_rxn_smarts': 'rxn_smarts'})
    else:
        df = df.rename(columns={'smarts_string': 'rxn_smarts'})
    df[cols].to_csv(Options.output_file, index=False)
    # reaction_id,substrate_id,diameter,direction,smarts_id,rxn_smarts,validated
    return


if __name__ == "__main__":
    Options = get_args()
    main()
