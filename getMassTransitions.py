#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import argparse
import sqlite3
import numpy as np
import pandas as pd

import gizmos


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('metanetx_file', help='chem_prop.tsv')
    parser.add_argument('retrorules_sql_file')
    parser.add_argument('output_file')
    parser.add_argument('--monoisotopic_mass', '-m', default=False, action='store_true', required=False, help='Flag. Use if metanetx_file has a "mm" column.')
    parser.add_argument('-d', '--decimals', required=False, type=int, default=2, help='Number of decimals kept. Default: 2.')
    parser.add_argument('--verbose', '-v', default=False, action='store_true', required=False)
    return parser.parse_args()


def write_output(reaction_id, mass_transition_round, mass_transition, substrate_id, substrate_mnx_id, substrate_mm, product_id, product_mnx_id, product_mm):
    line = ','.join([reaction_id, mass_transition_round, mass_transition, substrate_id, substrate_mnx_id, substrate_mm, product_id, product_mnx_id, product_mm])
    with open(Options.output_file, 'a') as f:
        f.write(line)
        f.write('\n')
    return


#################
# MAIN PIPELINE #
#################
def main():
    gizmos.print_milestone('Loading MetaNetX database...', Options.verbose)

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
        gizmos.print_milestone('Calculating monoisotopic mass...', Options.verbose)
        mnx_df['mm'] = mnx_df.SMILES.apply(gizmos.get_mm_from_str, is_smarts=False, rounded=False)
        fname = Options.metanetx_file[:-4] + '.mm_treated.tsv'
        mnx_df.to_csv(fname, index=True, sep='\t')

    gizmos.print_milestone('Querying RetroRules database...', Options.verbose)
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
    header = ','.join(['reaction_id', 'mass_transition_round', 'mass_transition', 'substrate_id', 'substrate_mnx_id', 'substrate_mm', 'product_id', 'product_mnx_id', 'product_mm'])
    with open(Options.output_file, 'w') as f:
        f.write(header)
        f.write('\n')

    gizmos.print_milestone('Identifying transitions...', Options.verbose)
    # rule_products
    # product_id, reaction_id, substrate_id, diameter, isStereo, stochiometry
    query1 = 'SELECT substrate_id, product_id FROM rule_products WHERE diameter=16 and isStereo=0 and reaction_id=?'
    query2 = 'SELECT mnxm FROM chemical_species WHERE id=?'
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
                             str(product_id), product_mnx_id, str(product_mm))

    return


if __name__ == "__main__":
    Options = get_args()
    main()
