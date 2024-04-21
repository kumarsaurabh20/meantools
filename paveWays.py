#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import sys
import glob
import argparse
import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
import svgutils.transform as sg
from svgelements import *

import gizmos
import svg

DrawingOptions.atomLabelFontSize = 80
DrawingOptions.dotsPerAngstrom = 100
DrawingOptions.bondLineWidth = 2.5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp','--structure_predictions', default='', action='store', required=True, help='Output from heraldPathways.py')
    parser.add_argument('-of','--output_folder', default='', required=True, action='store', help='All the output files will be saved in this folder!')
    parser.add_argument('-r','--reactions', default='', action='store', required=False, help='Output from heraldPathways.py')
    parser.add_argument('-praf','--pfam_RR_annotation', default='', action='store', required=False, help='Nine-column csv. reaction_id, uniprot_id, Pfams, KO, rhea_id_reaction, kegg_id_reaction, rhea_confirmation, kegg_confirmation, KO_prediction. Using this input will add to the pathway figures which pfams are capable of each reaction.')
    parser.add_argument('-gaf','--gene_annotation', default='', action='store', required=False, help='Two-column csv. Gene, pfam1;pfam2 Using this input in conjunction with pfam_RR_annotation_file will add to the pathway figures which uncorrelated genes in the genome are capable of each reaction.')
    parser.add_argument('-rr','--reaction_rules', default='strict', required=False, choices=['strict', 'medium', 'loose'], help='Default: strict.')
    parser.add_argument('-pdg','--use_pd_graph', default=True, required=False, choices=[True, False], type=bool, help='Default: True. Use False with older versions of networkx that do not have pandas compatability.')
    parser.add_argument('-pam','--print_all_molecules', default=False, choices=[True, False], type=bool, required=False, help='Flag. Use to generate SVGs of all molecules in the results.')
    parser.add_argument('-pup','--print_uncorr_genes', default=False, choices=[True, False], type=bool, required=False, help='Flag. Generate SVGs for all uncorrelated PFAMs.')
    parser.add_argument('-p','--pathway', default='', required=False, help='Input comma-separated nodes to generate SVGs of a specific pathway.')
    parser.add_argument('-v','--verbose', default=False, action='store_true', required=False)
    return parser.parse_args()


def get_structure_network(reactions_df):
    structures_network_df = reactions_df[['predicted_substrate_id', 'predicted_product_id']].drop_duplicates().copy()
    if 'correlation_substrate' in reactions_df:
        structures_network_df = structures_network_df.apply(get_structure_network_edge_info, df=reactions_df, axis=1)
    else:
        structures_network_df['genes_corr_substrate'] = 0
        structures_network_df['genes_corr_product'] = 0
        structures_network_df['genes_corr_both'] = 0
    # ++ genes_corr_substrate, genes_corr_product, genes_corr_both

    structures_network = gizmos.df_to_graph(structures_network_df, Options.use_pd_graph)
    # print(structures_network_df)
    #    predicted_substrate_id predicted_product_id genes_corr_substrate genes_corr_product genes_corr_both
    #root
    #NP_27475 NP_27475 VM_3137934 5 5 5
    #NP_27475 NP_27475 VM_7461151 2 2 2

    # print(structures_network)
    #DiGraph with 26 nodes and 25 edges
    return structures_network, structures_network_df


# Kumar
# Provides info on genes correlated with substrate, product, and both.
def get_structure_network_edge_info(cur_edge, df):
    substrate_mask = cur_edge.predicted_substrate_id == df['predicted_substrate_id']
    product_mask = cur_edge.predicted_product_id == df['predicted_product_id']
    # CORRELATED GENES
    cur_edge['genes_corr_substrate'] = len(
        df[substrate_mask & product_mask].dropna(subset=['correlation_substrate']))
    cur_edge['genes_corr_product'] = len(
        df[substrate_mask & product_mask].dropna(subset=['correlation_product']))
    cur_edge['genes_corr_both'] = len(
        df[substrate_mask & product_mask].dropna(subset=['correlation_substrate', 'correlation_product']))

    return cur_edge


def get_structure_network_attributes(root_id, structures_network, structures_df):
    temp_df = structures_df.set_index('predicted_substrate_id')
    reaction_distance_df = pd.DataFrame.from_dict(nx.single_source_shortest_path_length(structures_network.to_undirected(), root_id), orient='index', columns=['root_distance'])
    structures_attributes_df = pd.merge(temp_df, reaction_distance_df, left_index=True, right_index=True)
    structures_attributes_df.reset_index(inplace=True)
    structures_attributes_df.rename(columns={'index': 'predicted_id', 'predicted_substrate_mm':'predicted_mm', 'predicted_substrate_smiles': 'predicted_smiles'}, inplace=True)
    return structures_attributes_df

# Kumar 30/11/2022
# Instead of structures_df, it makes more sense to use cur_structure_attributes_df
# Also this folder should be generated inside root folder not outside the cur_root
def print_svg_molecules(cur_structure, path):
    '''
    cur_structure: cur_root_structures_attributes_df file 
           predicted_id predicted_smiles predicted_mm root_distance
       0   NP_152307 CCCCCCCCC=CCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCCN)... 855.67 0
       1   VM_4516106 CCCCCCCCC=CCCCCCCCCCCCC(=O)OC(COC(=O)CCCCCCCCC... 857.69 1
    path: path of the folder inside root_folder
    '''
    
    fname = os.path.join(path, cur_structure.predicted_id)
    if not os.path.exists(fname):
        try:
            mol = Chem.MolFromSmiles(cur_structure.predicted_smiles)
        except:
            print("WARNING :: Inaccurate SMILE for {}".format(cur_structure.predicted_id))
    if bool(mol):
        Draw.MolToFile(mol, fname + ".svg")
    
    return

#Kumar 17/01/23
#importing functions from the svg.py utilities
#Previous implementation was not based on SVG generated from RDkit
#Hence is the workaround
#Purging this method. If needed copy it from the svg_parser.py file.
#def get_mol_svg_lines(fname, cur_y):
def print_pathway(nodes, reactions_df, output_file, molecules_folder, print_reaction_data, print_uncorr_pfams=False, print_uncorr_genes=False, enzyme_df=pd.DataFrame()):

    # INITIATILIZING
    molecule_files = []
    text_annotation = []
    cur_y = 0
    cur_x = 0
    next_molecule_offset = 0
    text2 = []
    line_coordinates = []
    #line1 = sg.LineElement([(20,20),(20,50)], width=2, color="black")
    #line2 = sg.LineElement([(20,50),(25,45)], width=2, color="black")
    #line3 = sg.LineElement([(20,50),(15,45)], width=2, color="black")

    for i, cur_mol in enumerate(nodes):
        mol_file = os.path.join(molecules_folder, cur_mol + '.svg')
        
        # Kumar
        # some smiles have kekulizing errors so can not be drawn. Therefore it is essential to skip those molecules for which
        # there is no SVG file in the structures folder

        if os.path.exists(mol_file):
            molecule_files.append(mol_file)
            width, height = svg.get_size(mol_file)
            # MOL NAME
            if next_molecule_offset > 0:
                cur_x = 0
                cur_y = next_molecule_offset
                line_x = width/2
                line_coordinates.append(sg.LineElement([(line_x, next_molecule_offset - 0.2*next_molecule_offset),(line_x, next_molecule_offset + 0.2*next_molecule_offset)], width=3, color="black"))
                line_coordinates.append(sg.LineElement([(line_x, next_molecule_offset + 0.2*next_molecule_offset), (line_x + 5, next_molecule_offset + 0.2*next_molecule_offset - 5)], width=3, color="black"))
                line_coordinates.append(sg.LineElement([(line_x, next_molecule_offset + 0.2*next_molecule_offset), (line_x - 5, next_molecule_offset + 0.2*next_molecule_offset - 5)], width=3, color="black"))
            
            cur_x += 15
            cur_y += 50
            
            text2.append(sg.TextElement(cur_x, cur_y, cur_mol, size=15, weight="bold"))


            # MASS NAME
            substrate_mask = reactions_df.predicted_substrate_id == cur_mol
            product_mask = reactions_df.predicted_product_id == cur_mol
            ms_substrates = reactions_df.ms_substrate[substrate_mask].unique()
            ms_products = reactions_df.ms_product[product_mask].unique()
            ms_substrates = list(set(ms_substrates).union(set(ms_products)))
            ms_substrates = ' | '.join(ms_substrates)
            # text.append(ms_substrates)
            cur_y += 15
            text2.append(sg.TextElement(cur_x, cur_y, ms_substrates, size=15))
            cur_y += height * 0.75
            cur_x = width * 0.5 + 10
            # REACTION DATA
            if print_reaction_data:
                if i == len(nodes) - 1:  # last molecule
                    continue
                else:
                    product = nodes[i + 1]

                    substrate_mask = reactions_df.predicted_substrate_id == cur_mol
                    product_mask = reactions_df.predicted_product_id == product

                    cur_reaction_rows = reactions_df[substrate_mask & product_mask]

                    # GENE
                    if 'correlation_substrate' in reactions_df:
                        cur_reaction_rows = cur_reaction_rows[
                            ['gene', 'enzyme_pfams', 'correlation_substrate', 'correlation_product']].drop_duplicates()
                        cur_reaction_rows = cur_reaction_rows.sort_values(by=['enzyme_pfams', 'gene'])

                        for cur_gene in cur_reaction_rows.gene.unique():
                            gene_mask = cur_reaction_rows.gene == cur_gene
                            cur_gene_df = cur_reaction_rows[gene_mask]
                            cur_gene_pfams = cur_gene_df.enzyme_pfams.iloc[0]

                            subs_none_corr = cur_gene_df.correlation_substrate.isna()
                            prod_none_corr = cur_gene_df.correlation_product.isna()
                            subs_pos_corr = cur_gene_df.correlation_substrate > 0
                            subs_neg_corr = cur_gene_df.correlation_substrate < 0
                            prod_pos_corr = cur_gene_df.correlation_product > 0
                            prod_neg_corr = cur_gene_df.correlation_product < 0

                            if subs_none_corr.all():  # no corr
                                pass
                            elif subs_pos_corr.all():  # green ^
                                text2.append(sg.TextElement(cur_x, cur_y, "^", size=15, color="green"))
                            elif subs_neg_corr.all():  # red ^
                                text2.append(sg.TextElement(cur_x, cur_y, "^", size=15, color="red"))
                            else:  # blue ^
                                text2.append(sg.TextElement(cur_x, cur_y, "^", size=15, color="blue"))

                            if prod_none_corr.all():  # no corr
                                pass
                            elif prod_pos_corr.all():  # green v
                                text2.append(sg.TextElement(cur_x + 10, cur_y, "v", size=15, color="green"))
                            elif prod_neg_corr.all():  # red v
                                text2.append(sg.TextElement(cur_x + 10, cur_y, "v", size=15, color="red"))
                            else:  # blue v
                                text2.append(sg.TextElement(cur_x + 10, cur_y, "v", size=15, color="blue"))

                            cur_line = ' | '.join([cur_gene, cur_gene_pfams])
                            text2.append(sg.TextElement(cur_x + 20, cur_y, cur_line, size=15))
                            cur_y += 15

                    elif print_uncorr_pfams:
                        cur_reaction_ids = cur_reaction_rows.reaction_id.unique()
                        reaction_ids_mask = enzyme_df.reaction_id.isin(cur_reaction_ids)
                        cur_reaction_pfams = enzyme_df.uniprot_enzyme_pfams[reaction_ids_mask].unique()
                        if len(cur_reaction_pfams):
                            cur_reaction_pfams = {n for sub in cur_reaction_pfams for n in sub.split(';')}
                            cur_reaction_pfams = sorted(list(cur_reaction_pfams))
                            for cur_pfam in cur_reaction_pfams:
                                text2.append(sg.TextElement(cur_x, cur_y, cur_pfam, size=20))
                                cur_y += 15
                        else:
                            pass

                    elif print_uncorr_genes:
                        cur_reaction_ids = cur_reaction_rows.reaction_id.unique()
                        reaction_ids_mask = enzyme_df.reaction_id.isin(cur_reaction_ids)
                        cur_reaction_genes_df = enzyme_df[reaction_ids_mask]
                        cur_reaction_genes_df = cur_reaction_genes_df[['gene', 'enzyme_pfams']].drop_duplicates()
                        if not cur_reaction_genes_df.empty:
                            cur_reaction_genes_df = cur_reaction_genes_df.sort_values(by=['enzyme_pfams', 'gene'])
                            for j, cur_gene in cur_reaction_genes_df.iterrows():
                                cur_line = ' | '.join([cur_gene.gene, cur_gene.enzyme_pfams])
                                text2.append(sg.TextElement(cur_x, cur_y, cur_line, size=20))
                                cur_y += 15
                    else:
                        pass
            else:
                pass

                # ARROW
            if i == len(nodes) - 1:  # last molecule
                continue
            else:
                pass

            next_molecule_offset += 300

        else:
            pass

    svgs = svg.files_to_svg_dict(molecule_files)
    file_lists = list(svgs)
    reference = list(svgs)[0]
    svg.rescale(svgs)
    svg.change_positions(svgs)
    full_width = 2 * svgs[reference].width
    full_height = sum([svgs[i].height for i in file_lists])
    fig = sg.SVGFigure(full_width, full_height)
    fig.append([s.data for s in svgs.values()])
    fig.append(text2)
    fig.append(line_coordinates)
    fig.save(output_file)

    return

def print_all_pathways(pathway_nodes, cur_root_structures_attributes_df, reactions_df, enzyme_df, structures_df, out_folder):

    #structures_df.loc[pathway_nodes].apply(print_svg_molecules, out_folder=Options.svg_folder, axis=1)
    # Kumar 30/11/2022
    # There is no column with the name 'pathway_nodes'
    # cur_root_structures_attributes_df.apply(print_svg_molecules, out_folder=Options.svg_folder, axis=1)
    # cur_root_structures_attributes_df
    # predicted_id predicted_smiles predicted_mm root_distance
    # Generate all predicted_smiles in a structure folder with in root

    st_path = os.path.join(out_folder, 'structures/')
    if not os.path.exists(st_path):
        os.makedirs(st_path)

    cur_root_structures_attributes_df.apply(print_svg_molecules, path=st_path, axis=1)
    if 'correlation_substrate' in reactions_df:
        fname = os.path.join(out_folder, 'longest_path_root_genes.svg')
        print_pathway(pathway_nodes, reactions_df, fname, st_path, True)

    fname = os.path.join(out_folder, 'longest_path_root.svg')
    print_pathway(pathway_nodes, reactions_df, fname, st_path, False)

    if Options.print_uncorr_genes == True:
        fname = os.path.join(out_folder, 'longest_path_root_uncorr_pfams.svg')
        print_pathway(pathway_nodes, reactions_df, fname, st_path, True, Options.print_uncorr_pfams, enzyme_df=enzyme_df)
        fname = os.path.join(out_folder, 'longest_path_root_uncorr_genes.svg')
        print_pathway(pathway_nodes, reactions_df, fname, st_path, True, Options.print_uncorr_genes, enzyme_df=enzyme_df)
    return


def get_rooted_semiforward_network(structure_network_df, structures_attributes_df):
    """
    Prepares network so
    :param structure_network_df:
    :param structures_attributes_df:
    :return:
    """
    # todo check attributes
    structures_attributes_df2 = structures_attributes_df[['predicted_id', 'root_distance']]
    structure_network_df = structure_network_df.copy()

    structures_attributes_df2 = structures_attributes_df2.rename(columns={'root_distance': 'root_distance_substrate'})
    structures_attributes_df2.set_index('predicted_id')
    structure_network_df = pd.merge(structures_attributes_df2, structure_network_df, right_on='predicted_substrate_id', left_on='predicted_id', how='left')

    structure_network_df = structure_network_df.set_index('predicted_product_id').copy()
    structures_attributes_df2 = structures_attributes_df2.rename(columns={'root_distance_substrate': 'root_distance_product'})
    structure_network_df = pd.merge(structures_attributes_df2, structure_network_df, right_on='predicted_product_id', left_on='predicted_id')
    structure_network_df['semiforward_reaction'] = (structure_network_df.root_distance_product >= structure_network_df.root_distance_substrate)

    semiforward_network_df = structure_network_df[structure_network_df.semiforward_reaction]
    semiforward_network = gizmos.df_to_graph(semiforward_network_df, True)
    return semiforward_network


#################
# MAIN PIPELINE #
#################
def main():
    # OUTPUT INIT
    gizmos.log_init(Options)
    if not os.path.exists(Options.svg_folder):
        os.makedirs(Options.svg_folder)

    # ms_substrate, predicted_substrate_id, predicted_substrate_mm, predicted_substrate_smiles, InChI, reacted, root
    gizmos.print_milestone('Loading structures...', Options.verbose)
    structures_df = pd.read_csv(Options.structure_predictions, index_col=0)


    # Kumar
    # is_smart=False
    # without False the method get_mm_from_str is not caclulating mm for specific smiles; with False attribute its working fine.
    # Check this issue later
    if 'predicted_substrate_mm' not in structures_df:
        structures_df['predicted_substrate_mm'] = structures_df.predicted_substrate_smiles.apply(gizmos.get_mm_from_str,
                                                                                                 is_smarts=False)
    #if not Options.reactions or Options.print_all_molecules:
    #    structures_df.apply(print_svg_molecules, path=Options.svg_folder, axis=1)

    if Options.reactions:
        gizmos.print_milestone('Loading reactions...', Options.verbose)
        # INPUT
        # ms_substrate, ms_product, expected_mass_transition, predicted_mass_transition, mass_transition_difference,
        #  reaction_id, substrate_id, product_id, substrate_mnx, product_mnx, predicted_substrate_id,
        #   predicted_product_id, predicted_substrate_smiles, predicted_product_smiles, smarts_id, RR_substrate_smarts,
        #    RR_product_smarts, uniprot_id, uniprot_enzyme_pfams, KO, rhea_id_reaction, kegg_id_reaction,
        #     rhea_confirmation, kegg_confirmation, KO_prediction, gene, enzyme_pfams, correlation_substrate,
        #      P_substrate, correlation_product, P_product
        reactions_df = pd.read_csv(Options.reactions, index_col=None, dtype={'rhea_id_reaction': str,
                                                                                  'reaction_id': str})
        reactions_df = reactions_df.set_index('root')

        if Options.gene_annotation and not Options.pfam_RR_annotation:
            sys.exit('ERROR: Gene annotations input is only usable in combination with the pfam_RR_annotation_file input.')
        else:
            enzyme_df = gizmos.load_enzyme_input(Options)
            if Options.pfam_RR_annotation:
                Options.print_uncorr_pfams = True
                if Options.gene_annotation:
                    Options.print_uncorr_genes = True

        if Options.pathway:
            gizmos.print_milestone('Printing selected pathway...', Options.verbose)
            pathway = Options.pathway.split(',')
            print_all_pathways(pathway, reactions_df, enzyme_df, structures_df, Options.output_folder)
        else:
            gizmos.print_milestone('Generating networks...', Options.verbose)
            
            structures_network, structure_network_df = get_structure_network(reactions_df)
            fname = os.path.join(Options.output_folder, 'full_structure_network.csv')
            #structure_network_df.to_csv(fname, index=False)

            #nx.draw(structures_network, with_labels=True)
            #plt.show()
            
            # SUB NETWORKS
            subnetworks = [n for n in nx.connected_components(structures_network.to_undirected())]
            
            #for component in nx.connected_components(structures_network.to_undirected()):
                
            #    print("#### subnetwork ####")
            #    print(component)
            #    subgraph = structures_network.subgraph(component)
            #    nx.draw(subgraph, with_labels=True, node_color='green', edge_color="gray", node_size=50)
            #    plt.show()
            #    print("\n")
            
            ##  A connected component is a subset of nodes where:
            ##   1. Every node in the subset has a path to every other node
            ##   2. No node outside the subset has a path to a node in the subset
            ## Something like this, multiple connected components:
            ##  {'J', 'I', 'G', 'H', 'F'}
            ##   {'N', 'M', 'O', 'K', 'L'}
            ##   {'D', 'B', 'C', 'E', 'A'}

            subnetworks_dict = dict()
            for sub_idx, cur_sub in enumerate(subnetworks):
                for node in cur_sub:
                    subnetworks_dict[node] = sub_idx

            # PROCESS EACH ROOT NETWORK
            gizmos.print_milestone('Processing each network...', Options.verbose)
            for cur_root in reactions_df.index.unique():
                cur_root_out_folder = os.path.join(Options.output_folder, 'root_networks', cur_root + '/')
                if not os.path.exists(cur_root_out_folder):
                    os.makedirs(cur_root_out_folder)

                # GET NETWORK COMPONENT WITH ROOT
                cur_root_structures = subnetworks[subnetworks_dict[cur_root]]
                cur_root_structures_network = structures_network.subgraph(cur_root_structures)


                # GET ROOT-SPECIFIC DATA
                # Kumar
                # Structures_df is indexed with substrate_id and not the predicted molecules ??????
                #cur_root_structures_df = structures_df[structures_df.index.isin(cur_root_structures)].copy()
                cur_root_structures_df = structures_df[structures_df['root'].isin(cur_root_structures)]
                #cur_root_structures_df.to_csv("cur_root_structures_df.csv", index=False)
                subs_mask = structure_network_df.predicted_substrate_id.isin(cur_root_structures)
                prod_mask = structure_network_df.predicted_product_id.isin(cur_root_structures)
                cur_root_structures_network_df = structure_network_df[subs_mask & prod_mask]
                
                # Kumar
                # Feb 2024
                # adds edge score and avg. edge weights properties of each reactant
                cur_root_structures_network_df['gene_support'] = cur_root_structures_network_df.apply(gizmos.count_edge_support_summary, axis=1)
                cur_root_structures_network_df['avg_subs_edge_weight'] = cur_root_structures_network_df.apply(gizmos.get_means_for_substrate_and_product, df=reactions_df, substrate=True, axis=1)
                cur_root_structures_network_df['avg_prod_edge_weight'] = cur_root_structures_network_df.apply(gizmos.get_means_for_substrate_and_product, df=reactions_df, substrate=False, axis=1)


                # PREPARE STRUCTURES
                # Not sure why Hern used root values as indexers directly
                # Also what is the scope of the following 3 lines.
                # WHY ?? This value is always 0 - Kumar (28/11/2022) 
                #cur_root_structures_df.reset_index(drop=True, inplace=True)
                
                #cur_root_structures_df['root_mm'] = cur_root_structures_df.predicted_substrate_mm.loc[cur_root]
                #cur_root_structures_df['root_mm'] = cur_root_structures_df.loc[cur_root_structures_df['root'] == cur_root, 'predicted_substrate_mm']
                #cur_root_structures_df['root_mm_diff'] = cur_root_structures_df.predicted_substrate_mm - cur_root_structures_df.root_mm
                #cur_root_structures_df['root_mm_diff_abs'] = cur_root_structures_df.root_mm_diff.apply(abs)


                # GET STRUCTURE NETWORK ATTRIBUTES
                #cur_root_structures_attributes_df = get_structure_network_attributes(cur_root, cur_root_structures_network, structures_df, reactions_df)
                cur_root_structures_attributes_df = get_structure_network_attributes(cur_root, cur_root_structures_network, structures_df)
                cur_root_structures_attributes_df = cur_root_structures_attributes_df.drop_duplicates(subset=['predicted_id', 'root', 'root_distance'])
                #cur_root_structures_attributes_df.to_csv("cur_root_structures_attributes_df_dedup.csv", index=False)

                # OUTPUT
                fname = os.path.join(cur_root_out_folder, 'structure_network.csv')
                cur_root_structures_network_df.to_csv(fname, index=False)

                fname = os.path.join(cur_root_out_folder, 'structure_network_attributes.csv')
                cur_root_structures_attributes_df.to_csv(fname, index=True)

                # FORWARD DF
                # todo why?
                # Kumar 30/11/2022
                # This step is not adding new information to the final out put. So purged at the moment.
                #semiforward_network = get_rooted_semiforward_network(cur_root_structures_network_df, cur_root_structures_attributes_df)

                # Detect and remove cycles
                def remove_cycles(graph):
                    cycles = list(nx.simple_cycles(graph))
                    if cycles:
                        cycle = cycles[0]
                        gizmos.print_milestone(f'Cycle found: {cycle}', Options.verbose)
                        # If a cycle is found, remove one edge from the cycle
                        edge_to_remove = (cycle[0], cycle[1])
                        graph.remove_edge(*edge_to_remove)
                        gizmos.print_milestone(f'Edge removed: {edge_to_remove}', Options.verbose)
                    else:
                        gizmos.print_milestone("No cycles found.", Options.verbose)

                # MAKE DAG
                root_dag_structures_network = gizmos.get_dag_from_structures_network(cur_root_structures_network, cur_root_structures_attributes_df)  
                
                #while True:
                #    remove_cycles(root_dag_structures_network)
                #    if not list(nx.simple_cycles(root_dag_structures_network)):
                #        break

                longest_path = nx.dag_longest_path(root_dag_structures_network)  # list of nodes, ordered


                # OUTPUT MOLECULE SVGs
                print_all_pathways(longest_path, cur_root_structures_attributes_df, reactions_df, enzyme_df, structures_df, cur_root_out_folder)

    return


if __name__ == "__main__":
    Options = get_args()
    Options.log_file = os.path.join(Options.output_folder, 'log.txt')
    Options.correlation_file = ''
    Options.svg_folder = os.path.join(Options.output_folder, 'molecules/')
    main()


    ##### Remove ######
# Kumar 30/11/2022
# refactored the whole function
# it was difficult to get to the exact return format with just structres_df file
# Perhaps the problem was, instead of using reaction_df, the function was using structure_df
# Because root/predicted_substrate_id and predicted_product_id are separate columns and just with
# structures_df file it was not possible to extract smiles for the whole graph.
def get_structure_network_attributes_temp(root_id, structures_network, structures_df, reactions_df):
    reaction_distance_df = pd.DataFrame.from_dict(nx.single_source_shortest_path_length(structures_network.to_undirected(), root_id), orient='index', columns=['root_distance'])
    reaction_distance_mod_df = reaction_distance_df.reset_index().rename(columns={'index': 'nodes'})
    list_index = list(reaction_distance_mod_df['nodes'])
    reactions_df_mod = reactions_df.reset_index()
    temp_df = reactions_df_mod[reactions_df_mod.isin(list_index).any(axis=1)]
    substrate = temp_df[['predicted_substrate_id', 'predicted_substrate_smiles']].drop_duplicates('predicted_substrate_id').rename(columns={'predicted_substrate_id': 'predicted_id', 'predicted_substrate_smiles': 'predicted_smiles'})
    substrate['predicted_mm'] = substrate.predicted_smiles.apply(gizmos.get_mm_from_str, is_smarts=True)
    product = temp_df[['predicted_product_id', 'predicted_product_smiles']].drop_duplicates('predicted_product_id').rename(columns={'predicted_product_id': 'predicted_id', 'predicted_product_smiles': 'predicted_smiles'})
    product['predicted_mm'] = product.predicted_smiles.apply(gizmos.get_mm_from_str, is_smarts=True)
    structures_attributes_df = pd.concat([substrate, product], ignore_index=True)
    structures_attributes_dist_df = pd.merge(reaction_distance_mod_df, structures_attributes_df, how="left", left_on='nodes', right_on='predicted_id')
    structures_attributes_dist_df = structures_attributes_dist_df.drop(columns="nodes")
    structures_attributes_dist_df = structures_attributes_dist_df[['predicted_id', 'predicted_smiles', 'predicted_mm', 'root_distance']]
    return structures_attributes_dist_df
#####################
