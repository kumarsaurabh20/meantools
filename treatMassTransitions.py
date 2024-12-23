#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import argparse
import os
import pandas as pd
import seaborn as sns


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mass_transitions_file',
                        help='Output from getMassTransitions.py')
    parser.add_argument('output_folder')
    parser.add_argument('-d', '--decimals',
                        required=False,
                        type=int,
                        default=2,
                        help='Number of decimals kept. Default: 2.')
    return parser.parse_args()


#################
# MAIN PIPELINE #
#################
def main():
    # INPUT
    df = pd.read_csv(Options.mass_transitions_file, index_col=None)
    df['mass_transition_round'] = df.mass_transition.apply(round, args=(Options.decimals,))
    df = df[['reaction_id', 'mass_transition_round', 'mass_transition', 'substrate_id', 'substrate_mnx_id', 'substrate_mm',
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
    if not os.path.exists(Options.output_folder):
        os.makedirs(Options.output_folder)

    fname = os.path.join(Options.output_folder, 'mass_transitions.rounded.csv')
    df.to_csv(fname, index=False)

    fname = os.path.join(Options.output_folder, 'mass_transitions_counts.csv')
    df2.to_csv(fname, index=False)

    # PLOT
    sns_plot = sns.scatterplot(data=df2, x='mass_transition', y='n_reactions')
    fig = sns_plot.get_figure()
    fname = os.path.join(Options.output_folder, 'transitions_degeneracy.png')
    fig.savefig(fname)

    return


if __name__ == "__main__":
    Options = get_args()
    main()
