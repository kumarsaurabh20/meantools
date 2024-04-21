import os
import sys
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FragmentMatcher import FragmentMatcher
from rdkit import rdBase
import sqlite3


# input requires a numpy series
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


def print_milestone(string, verbose):
    """
    Prints milestone if verbose.
    :param string:
    :param verbose:
    :return:
    """
    if verbose:
        print(string)
    return


def pd_to_set(entry, separator):
    """
    Converts strings "x separator x separator x separator x" into set()
    :param entry:
    :param separator:
    :return:
    """
    if pd.isnull(entry):
        return set()
    else:
        return set(entry.split(separator))


def pd_to_list(entry, separator):
    """
    Converts strings "x separator x separator x separator x" into list()
    :param entry:
    :param separator:
    :return:
    """
    if pd.isnull(entry):
        return []
    else:
        return entry.split(separator)


def string_to_set(string, separator):
    """
    Self-explanatory
    :param string:
    :param separator:
    :return:
    """
    return set(string.split(separator))


def set_to_string(s, separator):
    if s:
        return separator.join(s)
    else:
        return None


def get_heatmap_visual_params(n_rows, n_cols):
    # get the tick label font size
    fontsize_pt = 15
    dpi = 72.27

    # compute the matrix height in points and inches
    matrix_height_pt = fontsize_pt * n_rows
    matrix_height = matrix_height_pt / dpi
    matrix_width_pt = fontsize_pt * n_cols
    matrix_width = matrix_width_pt / dpi

    return matrix_width, matrix_height


def plot_annotations(fig, orig_ax, n_ticks, ticks_labels):
    ann_ax = fig.add_axes(orig_ax.get_position(), frameon=False)  # invisible plot for annotations
    ann_ax.yaxis.tick_right()  # ticks on the right for invis plot
    ann_ax.set_xticks([])  # hides xticks
    ann_ax.set_yticks(np.arange(n_ticks + 1))  # makes major yticks (to define y range)
    ann_ax.set_yticks([])  # hides major yticks
    ann_ax.set_yticks(np.arange(0.5, n_ticks + 0.5), minor=True)  # makes minor yticks
    ann_ax.set_yticklabels(ticks_labels, minor=True)  # makes minor yticks labels

    return fig


def import_from_sql(sqlite_db_name, sqlite_tablename="", df_columns=[], conditions={}, structures = False, clone = False):

    '''
    This function imports data from a SQLite database. The data can be fetched using a tablename. 
    Conditions is a dictionary here that has query elements. df_columns is a list of
    columns that goes immediately with the SELECT clause and also with the returning dataframe columnames.

    :param1 sqlite_db_name: Options object from the standard arguments block
    :param2 sqlite_tablename: sqlite correlation tablename
    :param3 conditions: example: conditions = {'gene': ['Solyc03g093590.1', 'Solyc11g008460.1', 'Solyc07g008140.2'], 'metabolite': ['M600T895', 'M274T851', 'M276T882']}
    :param4 df_columns: list of column names that have to retrieved: [metabolite, gene, correlations, P]
    :param5 structures: only enable when data from metabolite annotation table is required
    :param6 clone: Only enable when data from clusterONE output is required
    '''

    if os.path.isfile(sqlite_db_name):
        conn = sqlite3.connect(sqlite_db_name)
    else:
        raise FileNotFoundError("The {} file was not found! Did you run the correlation step first?".format(sqlite_db_name))


    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_db_name)
    
    if clone == True:
        
        # Get a list of table names from the database
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = conn.execute(tables_query).fetchall()

        # Filter tables that contain the keyword "clone"
        clone_tables = [table[0] for table in tables if 'clone' in table[0].lower()]

        # Create a dictionary to store DataFrames
        dfs = {}

        # Loop through each clone table and fetch data into a DataFrame
        for table_name in clone_tables:
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql_query(query, conn)
            dfs[table_name] = df

        return dfs

    else:

        if not bool(conditions):
            query = f"SELECT {', '.join(df_columns)} FROM {sqlite_tablename};"
        elif structures == True:
            query = f"SELECT {', '.join(df_columns)} FROM {sqlite_tablename} WHERE metabolite IN {tuple(conditions['metabolite'])};"
        else:
            query = f"SELECT {', '.join(df_columns)} FROM {sqlite_tablename} WHERE {' AND '.join([f'{key} IN {tuple(conditions[key])}' for key in conditions])};"

        # Execute the query and fetch the results
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        # Display or further process the extracted data
        result_df = pd.DataFrame(results, columns=df_columns)

        return result_df
    
    conn.close()


def export_to_sql(database, df, tablename, index=False):

    conn = sqlite3.connect(database)
    #cursor = conn.cursor()
    #if df.shape[0] > 5000:
    #   cursor.execute("PRAGMA page_size = 15000;")

    df.to_sql(tablename, conn, if_exists="append", index=index)
    conn.close()
    return


def log_init(Options):
    if not os.path.exists(Options.output_folder):
        os.makedirs(Options.output_folder)

    with open(Options.log_file, 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n\nArguments:\n')
        for arg, value in sorted(vars(Options).items()):
            f.write(str(arg) + ' ' + str(value) + '\n')
    return


def apply_MAD_filter(df, Options):
    print_milestone('Applying MAD filter...', Options.verbose)
    df['MAD'] = df.apply(calc_MAD, axis=1)

    mask = df.MAD > 0
    df_filtered = df[mask]

    del df_filtered['MAD']

    return df_filtered


def get_mass_range(mz, tolerance):
    """
    Gets mass range to query according to a tolerance in ppm.
    :param mz:
    :param tolerance:
    :return:
    """
    # 30 ppm of x = x * (30 / 1000000)
    tol = mz * (tolerance / 1000000)
    top = mz + tol
    btm = mz - tol
    return btm, top


def get_adduct_data(cur_mass_row, adducts_df, Options):
    """
    Gets possible mm's for the mz of cur_mass
    :param cur_mass_row:
    :param Options:
    :param adducts_df:
    :return:
    """
    def get_cur_mm_adducts(cur_adduct_row):
        mm_low = (cur_mass_row.mz_low - cur_adduct_row.Mass) / cur_adduct_row.Mult
        mm_high = (cur_mass_row.mz_high - cur_adduct_row.Mass) / cur_adduct_row.Mult
        mm = (cur_mass_row.mz - cur_adduct_row.Mass) / cur_adduct_row.Mult
        adduct_name = cur_adduct_row.name
        if mm <= 0:
            return pd.DataFrame()
        else:
            df = pd.DataFrame([[cur_mass_row.ms_name, cur_mass_row.mz, mm, mm_low, mm_high, adduct_name]],
                              columns=['ms_name', 'mz', 'observed_mm', 'mm_low', 'mm_high', 'adduct_name'])
            return df

    cur_mass_row['mz_low'], cur_mass_row['mz_high'] = get_mass_range(cur_mass_row.mz, Options.tolerance)

    results_df = adducts_df.apply(get_cur_mm_adducts, axis=1)
    results_df = pd.concat(results_df.values.tolist())

    return results_df


def get_subs_string(full_string):
    """
    Cleans reaction smarts/smiles into substrate and product.
    :param full_string:
    :return:
    """
    return full_string.split('>>')[0].strip('()')


def get_prod_string(full_string):
    """
    Cleans reaction smarts/smiles into substrate and product.
    :param full_string:
    :return:
    """
    return full_string.split('>>')[1].strip('()')


def generate_virtual_molecule(row, rxn_smarts_name, substrate_smiles_name, is_mol=False):
    """
    Applies reaction to substrate to obtain a list of possible products
    :param row:
    :param substrate_smiles_name: Name in row
    :param rxn_smarts_name: Name in row.
    :param is_mol:
    :return:
    """

    if is_mol:
        substrate_mol = row[substrate_smiles_name]
        rxn = row[rxn_smarts_name]
    else:

        # Kumar
        # Main step which predicts product (product, smiles, mm)
        # Uses rxn_smarts_name to estimate reaction
        # substrate from substrate smiles name and create substrate mol from substrate smile
        # Substrate mol is used to run as a reactant and predict products
        # product mol is converted to product smiles and product mm is then calculated. 
        reaction_smarts = row[rxn_smarts_name]
        rxn = Chem.ReactionFromSmarts(reaction_smarts)
        substrate_smiles = row[substrate_smiles_name]
        substrate_mol = Chem.MolFromSmiles(substrate_smiles)

    product_list = []
    smiles_list = []
    mm_list = []
    product_sets = rxn.RunReactant(substrate_mol, 0)
    for cur_product_set in product_sets:
        for cur_product in cur_product_set:
            cur_smiles = Chem.MolToSmiles(cur_product)
            cur_mm = get_mm_from_mol(cur_product, is_smarts=False)
            if cur_product in product_list or cur_smiles in smiles_list or np.isnan(cur_mm):
                continue
            else:     # identical product in list (p0 == p1), or empty list
                product_list.append(cur_product)
                smiles_list.append(cur_smiles)
                mm_list.append(cur_mm)

    if product_list:
        row['predicted_product_mol_list'] = product_list
        row['predicted_product_smiles_list'] = smiles_list
        row['predicted_product_mm_list'] = mm_list
    else:
        row['predicted_product_mol_list'] = []
        row['predicted_product_smiles_list'] = []
        row['predicted_product_mm_list'] = []

    return row


def remove_valence_from_smarts(smarts_str):
    clean_smarts_str = re.sub(r'&v\d+', '', smarts_str)
    return clean_smarts_str


def remove_valence_and_mergeh_from_smarts(smarts_str):
    clean_smarts_str = re.sub(r'&.*?(?=[:\]])', '', smarts_str)
    return clean_smarts_str


def get_mm_from_mol(mol, is_smarts, rounded=True):
    """
    Workaround for calculating the mass of mols from smarts that do not constitute actual molecules (which, because of
    wrong valence, we cannot use rdkit weight functions).
    :param mol:
    :param is_smarts:
    :param rounded:
    :return:
    """
    if is_smarts:
        return sum([atom.GetMass() for atom in mol.GetAtoms()])
    else:
        try:
            mol.UpdatePropertyCache()
            mm = Descriptors.ExactMolWt(mol)
            if rounded:
                return round(mm, 2)
            else:
                return mm
        except ValueError:                  # Raised when atoms end up with forbidden valences.
            return np.nan


def get_mm_from_str(mol_str, is_smarts, rounded=True):
    """
    Workaround for calculating the mass of mols from smarts that do not constitute actual molecules (which, because of
    wrong valence, we cannot use rdkit weight functions).
    :param mol_str:
    :param is_smarts:
    :param rounded:
    :return:
    """
    if not type(mol_str) == str or not mol_str:
        return np.nan
    if is_smarts:
        mol = Chem.MolFromSmarts(mol_str)
    else:
        mol = Chem.MolFromSmiles(mol_str)
    if bool(mol):
        if rounded:
            return get_mm_from_mol(mol, is_smarts)
        else:
            return get_mm_from_mol(mol, is_smarts, rounded=False)
    else:
        return np.nan


def get_unique_mols(df, unique_dict, use_mols):
    if len(df) == 1:
        unique_dict[df.index[0]] = True
    elif use_mols:
        for i, j in it.combinations(df.index, 2):
            mol_i = df.predicted_product_mol.loc[i]
            mol_j = df.predicted_product_mol.loc[j]
            if mol_i.HasSubstructMatch(mol_j) and mol_j.HasSubstructMatch(mol_i):   # then they are identical. i is rep
                unique_dict[j] = False
                if i not in unique_dict:
                    unique_dict[i] = True
            else:
                if i not in unique_dict:
                    unique_dict[i] = True
                if j not in unique_dict:
                    unique_dict[j] = True
    else:   # predicted_product_smiles
        unique_smiles_df = df.drop_duplicates(subset=['predicted_substrate_id', 'predicted_product_smiles'], keep='first')
        for i in df.index:
            if i in unique_smiles_df.index:
                unique_dict[i] = True
            else:
                unique_dict[i] = False
    return unique_dict


def match_pattern(mol, pattern_str):
    if pattern_str:
        p = FragmentMatcher()
        p.Init(pattern_str)
        return bool(p.HasMatch(mol))
    else:
        return False


def generate_new_ids(n, all_ids):
    new_ids = []
    while len(new_ids) < n:
        new = '_'.join(['VM', str(np.random.randint(0, 9999999)).zfill(7)])
        if new not in new_ids and new not in all_ids:
            new_ids.append(new)
    return new_ids


def get_next_rules(map_df, all_smarts_passed, all_smarts_tested):
    """
    Get next rules to query based on "smarts_is_in" (next steps), taking into account "smarts_has" (requirements) and
    which smarts have been queried before.
    :param map_df:
    :param all_smarts_passed:
    :param all_smarts_tested:
    :return:
    """
    rules_passed_mask = map_df.smarts_id.isin(all_smarts_passed)
    rules_passed_df = map_df[rules_passed_mask]

    if len(rules_passed_df.smarts_is_in):
        next_smarts = set.union(*rules_passed_df.smarts_is_in)      # list of sets -> set
    else:
        next_smarts = set()

    # Any smarts with a "smarts_has" need to pass all the "smarts_has" before testing.
    next_df = map_df[map_df.smarts_id.isin(next_smarts)].copy()
    next_df['requirements_met'] = next_df.smarts_has.apply(lambda x: x.issubset(all_smarts_passed))  # bool column
    next_df['to_remove'] = ~next_df.requirements_met                # remove smarts that do not meet requirements
    smarts_to_remove = set(next_df.smarts_id[next_df.to_remove])

    # remove smarts_to_remove, and remove rules that have been tested already
    next_smarts = next_smarts - smarts_to_remove
    next_smarts = next_smarts - all_smarts_tested
    return next_smarts


def filter_df_with_map(df, map_df, recent_smarts_passed, all_smarts_passed, all_smarts_tested):
    """
    Given a list of smarts, gets filters and calls "get_next_rules". Also updates all_smarts_passed, for proper usage
    of the next function.
    :param df:
    :param map_df:
    :param recent_smarts_passed:
    :param all_smarts_passed:
    :param all_smarts_tested:
    :return:
    """
    if not recent_smarts_passed:
        # We only enter this if in: -first iteration (query base rules)
        #                           -last iteration (query base rules => they have been queried => nothing to query)
        next_smarts_list = map_df.smarts_id[map_df.is_base]
    else:
        # First update recent_smarts_passed (smarts that passed test) to also have all identicals.
        identical_smarts = map_df.identity[map_df.smarts_id.isin(recent_smarts_passed)]
        if len(identical_smarts):
            identical_smarts = set.union(*identical_smarts)     # this expands list of sets into union set.
        else:
            identical_smarts = set()
        
        # We update all_smarts_passed and tested
        recent_smarts_passed.update(identical_smarts)
        all_smarts_tested.update(recent_smarts_passed)
        all_smarts_passed.update(recent_smarts_passed)
        # We get the next rules in the map
        next_smarts_list = get_next_rules(map_df, all_smarts_passed, all_smarts_tested)

    mask_to_test = df.smarts_id.isin(next_smarts_list)
    mask_not_tested = ~df.smarts_id.isin(all_smarts_tested)
    all_smarts_tested.update(next_smarts_list)      # Preemptive update, as they will be tested in the next steps.
    return df[mask_to_test & mask_not_tested].copy()


def query_filtered_rxn_db(rxn_df, map_df, base_rules_df, Options):
    """
    Queries each metabolite(metabolites_df) and finds which smiles(_db_df) are in it.
    :param rxn_df:
    :param map_df:
    :param base_rules_df:
    :param Options:
    :return: success_df
    """
    only_query_small = Options.only_query_small

    all_smarts_tested = set()
    all_smarts_passed = set()

    # Querying base rules
    all_smarts_tested.update(base_rules_df.smarts_id)
    cur_base_rules_df = base_rules_df.copy()
    cur_base_rules_df['predicted_substrate_id'] = rxn_df['predicted_substrate_id'].iloc[0]
    cur_base_rules_df['predicted_substrate_mol'] = rxn_df['predicted_substrate_mol'].iloc[0]

    cur_base_rules_df = cur_base_rules_df.apply(generate_virtual_molecule, axis=1,
                                                rxn_smarts_name='rxn', substrate_smiles_name='predicted_substrate_mol',
                                                is_mol=True)
    # ^^ this has added: predicted_product_mol_list, predicted_product_smiles_list, predicted_product_mm_list

    rules_passed_mask = cur_base_rules_df.predicted_product_mol_list.str.len().apply(bool)
    # ^^ bool series
    recent_smarts_passed = set(cur_base_rules_df.smarts_id[rules_passed_mask])

    success_df = pd.DataFrame()
    
    # Querying small rules
    while recent_smarts_passed:
        filtered_df = filter_df_with_map(rxn_df, map_df, recent_smarts_passed, all_smarts_passed, all_smarts_tested)
        if not filtered_df.empty:
            filtered_df_to_react = filtered_df[['rxn_smarts', 'rxn', 'predicted_substrate_id',
                                                'predicted_substrate_mol']].copy()
            filtered_df_to_react = filtered_df_to_react.drop_duplicates(subset=['rxn_smarts', 'predicted_substrate_id'])
            filtered_df_to_react = filtered_df_to_react.apply(generate_virtual_molecule, axis=1,
                                                              rxn_smarts_name='rxn',
                                                              substrate_smiles_name='predicted_substrate_mol',
                                                              is_mol=True)
            # ^^ this has added: predicted_product_mol_list, predicted_product_smiles_list, predicted_product_mm_list
            filtered_df_to_react = filtered_df_to_react.drop(columns=['rxn', 'predicted_substrate_mol'])    # for merge
            filtered_df = pd.merge(filtered_df, filtered_df_to_react, how='inner')
            del filtered_df_to_react  # memory management
            rules_passed_mask = filtered_df.predicted_product_smiles_list.str.len().apply(bool)
            # ^^ bool series
            recent_smarts_passed = set(filtered_df.smarts_id[rules_passed_mask])
            success_df = pd.concat([success_df, filtered_df[rules_passed_mask]])

        else:
            recent_smarts_passed = set()

    # Querying all rules
    if not only_query_small:
        
        # From all_smarts_passed we can find which reaction_substrates to query.
        smarts_ids_mask = rxn_df.smarts_id.isin(all_smarts_passed)
        reaction_substrates_to_query = rxn_df.reaction_substrate[smarts_ids_mask]
        reaction_substrates_to_query_mask = rxn_df.reaction_substrate.isin(reaction_substrates_to_query)

        # We have to remove all smarts that have been tested already.
        all_smarts_untested_mask = ~rxn_df.reaction_id.isin(all_smarts_tested)
        # apply
        filtered_df = rxn_df[reaction_substrates_to_query_mask & all_smarts_untested_mask].copy()
        if not filtered_df.empty:
            filtered_df_to_react = filtered_df[['rxn_smarts', 'rxn', 'predicted_substrate_id',
                                                'predicted_substrate_mol']].copy()
            filtered_df_to_react = filtered_df_to_react.drop_duplicates(subset=['rxn_smarts', 'predicted_substrate_id'])
            filtered_df_to_react = filtered_df_to_react.apply(generate_virtual_molecule, axis=1,
                                                              rxn_smarts_name='rxn',
                                                              substrate_smiles_name='predicted_substrate_mol',
                                                              is_mol=True)
            # ^^ this has added: predicted_product_mol_list, predicted_product_smiles_list, predicted_product_mm_list
            filtered_df_to_react = filtered_df_to_react.drop(columns=['rxn', 'predicted_substrate_mol'])    # for merge
            filtered_df = pd.merge(filtered_df, filtered_df_to_react, how='inner')
            del filtered_df_to_react  # memory management
            rules_passed_mask = filtered_df.predicted_product_smiles_list.str.len().apply(bool)
            filtered_df = filtered_df[rules_passed_mask]
            recent_smarts_passed = set(filtered_df.smarts_id)
            filtered_df = pd.merge(filtered_df, rxn_df)
            # ^^ bool df series
            success_df = pd.concat([success_df, filtered_df], sort=True)
        else:
            recent_smarts_passed = set()

    all_smarts_passed.update(recent_smarts_passed)
    all_smarts_tested.update(recent_smarts_passed)

    success_df = success_df.reset_index(drop=True)

    # expand predicted_product_smiles_list, predicted_product_mm_list and predicted_product_mol_list

    # Kumar
    # 15/09/2022
    # added a conditional
    if not success_df.empty:
        lens = [len(item) for item in success_df.predicted_product_smiles_list]
        smiles_flat = [n for sub in success_df.predicted_product_smiles_list for n in sub]
        mols_flat = [n for sub in success_df.predicted_product_mol_list for n in sub]
        mm_flat = [n for sub in success_df.predicted_product_mm_list for n in sub]
        new_df = pd.DataFrame({'predicted_product_smiles': smiles_flat,
                           'predicted_product_mol': mols_flat,
                           'predicted_product_mm': mm_flat},
                          index=np.repeat(success_df.index, lens))
        del success_df['predicted_product_smiles_list'], success_df['predicted_product_mm_list']
        del success_df['predicted_product_mol_list']
        success_df = pd.merge(success_df, new_df, left_index=True, right_index=True)

    # Check transitions
        success_df['predicted_mass_transition'] = success_df.predicted_product_mm - success_df.predicted_substrate_mm
        success_df['mass_transition_difference'] = abs(success_df.predicted_mass_transition -
                                                   success_df.expected_mass_transition)
        success_df = success_df[success_df.mass_transition_difference <= Options.max_mass_transition_diff]

        # select largest diameter per transitiion-reaction_substrate-product
        success_df = success_df.sort_values(by='diameter', ascending=False)
        if 'gene' in success_df.columns:
            success_df = success_df.drop_duplicates(subset=['predicted_substrate_smiles', 'predicted_product_smiles',
                                                        'reaction_id', 'substrate_id', 'gene'], keep='first')
        else:
            success_df = success_df.drop_duplicates(subset=['predicted_substrate_smiles', 'predicted_product_smiles',
                                                        'reaction_id', 'substrate_id'], keep='first')

        success_df = success_df.reset_index(drop=True)

        # Kumar
        # 19/09/2022
        fil_df = success_df.drop_duplicates(subset=['reaction_id', 'substrate_id', 'predicted_product_mm'])

        return fil_df
    else:
        return success_df



def conditional_mol(row_to_mol, is_rxn=False):
    """
    Mols only what has not been moled before.
    :param row_to_mol:
    :param is_rxn:
    :return:
    """
    if not is_rxn:
        col = 'predicted_substrate_mol'
    else:
        col = 'rxn'

    if pd.isnull(row_to_mol[col]):
        if not is_rxn:
            row_to_mol[col] = Chem.MolFromSmiles(row_to_mol['predicted_substrate_smiles'])
        else:
            row_to_mol[col] = Chem.ReactionFromSmarts(row_to_mol['rxn_smarts'])

    return row_to_mol


def df_to_graph(df, use_pd_graph):
    """
    Converts a dataframe to a networkx graph.

    Args:
        use_pd_graph:
        df (pd.DataFrame):

    Returns:
        nx.MultiGraph
    """
    if use_pd_graph:
        G = nx.from_pandas_edgelist(df, 'predicted_substrate_id', 'predicted_product_id', edge_attr=True, create_using=nx.DiGraph())
    else:
        G = nx.MultiDiGraph()
        for i, cur_row in df.iterrows():
            G.add_edge(cur_row.source, cur_row.target)

            #Note: loop below is not necessary? cur_row.index is the predicted_substrate_id
            for attribute in cur_row.index:
                G[cur_row.source][cur_row.target].update({attribute: cur_row[attribute]})

    return G


# This function gathers edge weight support for the network
def get_means_for_substrate_and_product(row, df, substrate=True):

    substrate_id = row['predicted_substrate_id']
    product_id = row['predicted_product_id']

    filtered_df = df[(df['predicted_substrate_id'] == substrate_id) & (df['predicted_product_id'] == product_id)]

    # Convert empty strings in numeric columns to NaN
    filtered_df[['correlation_substrate', 'P_substrate', 'correlation_product', 'P_product']] = \
        filtered_df[['correlation_substrate', 'P_substrate', 'correlation_product', 'P_product']].apply(pd.to_numeric, errors='coerce')

    if substrate:
        mean_correlation_substrate = filtered_df['correlation_substrate'].mean()
        return mean_correlation_substrate
    else:
        mean_correlation_product = filtered_df['correlation_product'].mean()
        return mean_correlation_product



def count_edge_support_summary(data):
    
    # use edge weights here from MR
    support = data['genes_corr_substrate'] + data['genes_corr_product']
    support -= 2 * data['genes_corr_both']

    return support


def get_edges_from_nodes(nodes):
    """
    return list of edges from ordered nodes.
    :param nodes:
    :return:
    """
    edges = []
    for i in range(len(nodes)):
        if i == len(nodes) - 1:  # last loop
            j = 0
        else:
            j = i + 1
        source = nodes[i]
        target = nodes[j]
        edges.append([source, target])
    return edges

# Used for generating DAG from the structures_network data structure
def get_dag_from_structures_network(structures_network, structures_attributes_df):
    # CLEAN CYCLES WITH 2 NODES
    # Kumar
    # Try None instead of 2 nodes to clean cycles
    #edges_to_remove = get_cycle_edges_to_remove(structures_network, structures_attributes_df, None)
    edges_to_remove = get_cycle_edges_to_remove(structures_network, structures_attributes_df, 2)
    dag = structures_network.copy()
    dag.remove_edges_from(edges_to_remove)

    # CLEAN OTHER CYCLES
    edges_to_remove = get_cycle_edges_to_remove(dag, structures_attributes_df)
    dag.remove_edges_from(edges_to_remove)
    return dag


def get_cycle_edges_to_remove(G, structures_attributes_df, cycle_size=0):
    edges_to_remove = []
    
    #nx.draw(G, with_labels=True, node_size=50)
    #plt.show()

    #structures_attributes_df_indexed = structures_attributes_df.set_index('root')

    for cur_cycle_nodes in nx.simple_cycles(G):  # generator; iterates over all cycles in the graph
        
        print(f"This is cur_cycle_nodes {cur_cycle_nodes}")

        if not cycle_size or len(cur_cycle_nodes) == cycle_size:
            edges_to_break_cycle = []
            cycle_supports = {}

            for substrate, product in get_edges_from_nodes(cur_cycle_nodes):

                print(f"This is substrate: {substrate} and this is product: {product}")


                cur_edge_data = G.get_edge_data(substrate, product)

                print(f"This is current_edge_data {cur_edge_data}")

                cur_edge_support = count_edge_support_summary(cur_edge_data)

                print(f"This is cur_edge_support {cur_edge_support}")
                
                # Kumar
                # remove duplicates in the structures_attributes_df file
                if substrate not in list(structures_attributes_df['root']) or product not in list(structures_attributes_df['predicted_id']):
                    
                    print(f"This substrate {substrate} and product {product} are not found in the structures_attributes_df_indexed.index")
                    pass

                else:

                    print(f"This substrate {substrate} and product {product} were found in the structures_attributes_df_indexed.index")

                    #cur_substrate_distance = structures_attributes_df_indexed.root_distance[substrate]
                    #cur_product_distance = structures_attributes_df_indexed.root_distance[product]

                    cur_substrate_distance = structures_attributes_df.loc[(structures_attributes_df['root'] == substrate), 'root_distance'].iloc[0]
                    cur_product_distance = structures_attributes_df.loc[(structures_attributes_df['predicted_id'] == product), 'root_distance'].iloc[0]
                    

                    print(f"This is cur_substrate_distance: {cur_substrate_distance}")
                    print(f"This is cur_product_distance: {cur_product_distance}")
                
                if cur_substrate_distance == cur_product_distance:
                    cycle_supports[substrate] = cur_edge_support
                elif cur_substrate_distance > cur_product_distance:     # this means the edge is towards the root
                    edges_to_break_cycle.append([substrate, product])

            if not edges_to_break_cycle:
                substrate = min(cycle_supports, key=cycle_supports.get)
                if substrate == cur_cycle_nodes[-1]:     # last reaction in cycle
                    product = cur_cycle_nodes[0]
                else:
                    product = cur_cycle_nodes[cur_cycle_nodes.index(substrate)+1]
                edges_to_remove.append([substrate, product])
            else:
                for n in edges_to_break_cycle:
                    edges_to_remove.append(n)
    print(f"These edges are going to be removed {edges_to_remove}")
    return edges_to_remove


def load_correlations(Options):
    """
    loads correlation dataframe
    :return:
    """
    # COEXPRESSION
    print_milestone('Loading coexpression...', Options.verbose)
    correlation_df = pd.read_csv(Options.correlation_file, index_col=None)
    correlation_df = correlation_df.rename(columns={correlation_df.columns[0]: 'ms_name',
                                                    correlation_df.columns[1]: 'gene',
                                                    correlation_df.columns[2]: 'correlation',
                                                    correlation_df.columns[3]: 'P'})

    if not Options.corr_cutoff == 0:
        correlation_df = correlation_df[abs(correlation_df['correlation']) >= Options.corr_cutoff]
    if not Options.corr_p_cutoff == 1:
        correlation_df = correlation_df[abs(correlation_df['P']) <= Options.corr_p_cutoff]

    return correlation_df


def load_enzyme_input(Options):
    """
    loads gene annotations, pfam-RR relationship file, and correlation and merges them.
    :return:
    """
    pfam_dict_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pfams_dict.csv')  # Acc,Name,Desc
    # GENE ANNOTATIONS
    # gene, pfam1;pfam2
    if Options.gene_annotation:
        print_milestone('Loading gene annotations...', Options.verbose)
        annotations_df = pd.read_csv(Options.gene_annotation, index_col=None)
        annotations_df = annotations_df.rename(columns={annotations_df.columns[0]: 'gene',
                                                        annotations_df.columns[1]: 'enzyme_pfams'})
        enzyme_pfams_list = annotations_df.enzyme_pfams.apply(pd_to_list, separator=';')
        # expand gene annotations so there's one pfam per line (but we keep the "pfam" annotation that have them all)
        lens = [len(item) for item in enzyme_pfams_list]
        new_df = pd.DataFrame({'gene': np.repeat(annotations_df.gene, lens),
                               'pfam_rule': np.concatenate(enzyme_pfams_list)})
        annotations_df = pd.merge(annotations_df, new_df, how='outer')
        del enzyme_pfams_list, new_df
    else:
        annotations_df = pd.DataFrame()

    # PFAM - RR
    # reaction_id, uniprot_id, Pfams, KO, rhea_id_reaction, kegg_id_reaction, rhea_confirmation, kegg_confirmation,
    #  KO_prediction
    if Options.pfam_RR_annotation:
        print_milestone('Loading pfam-RR annotations...', Options.verbose)
        pfam_rules_df = pd.read_csv(Options.pfam_RR_annotation, index_col=None, dtype={'rhea_id_reaction': str,
                                                                                            'KO': str,
                                                                                            'kegg_id_reaction': str})
        pfam_rules_df = pfam_rules_df.rename(columns={pfam_rules_df.columns[1]: 'uniprot_id',
                                                      pfam_rules_df.columns[2]: 'uniprot_enzyme_pfams_acc'})
        pfam_rules_df['reaction_id'] = pfam_rules_df.reaction_id.astype('str')

        # filter type of anotations (strict, medium, loose)
        if Options.reaction_rules == 'strict':
            pfam_rules_df = pfam_rules_df[pfam_rules_df.experimentally_validated]
        elif Options.reaction_rules == 'medium':
            pfam_rules_df = pfam_rules_df[pfam_rules_df.experimentally_validated |
                                          pfam_rules_df.Pfam_ec_prediction]
        else:  # loose
            pass  # they are all there

        # convert pfam_acc to pfam
        pfam_dict = pd.read_csv(pfam_dict_file, index_col=None)
        pfam_dict.index = pfam_dict.Acc.apply(lambda x: x.split('.')[0])  # Acc,Name,Desc

        uniprot_enzyme_pfams_acc_list = pfam_rules_df.uniprot_enzyme_pfams_acc.apply(pd_to_list, separator=' ')
        pfam_rules_df['uniprot_enzyme_pfams_list'] = [[pfam_dict.Name.loc[k] for k in row if k in pfam_dict.index]
                                                      for row in uniprot_enzyme_pfams_acc_list]
        pfam_rules_df['uniprot_enzyme_pfams'] = pfam_rules_df.uniprot_enzyme_pfams_list.apply(';'.join)

        # Expand df so there is only one pfam per row.
        lens = [len(item) for item in pfam_rules_df.uniprot_enzyme_pfams_list]
        pfams_flat = [n for sub in pfam_rules_df.uniprot_enzyme_pfams_list for n in sub]
        new_df = pd.DataFrame({'uniprot_id': np.repeat(pfam_rules_df.uniprot_id, lens),
                               'pfam_rule': pfams_flat}).drop_duplicates()

        pfam_rules_df = pd.merge(pfam_rules_df, new_df, how='outer')  # on uniprot_id.
        del pfams_flat, uniprot_enzyme_pfams_acc_list, new_df
        del pfam_rules_df['uniprot_enzyme_pfams_list'], pfam_rules_df['uniprot_enzyme_pfams_acc']
        # ++ pfam_rule, uniprot_enzyme_pfams >>> -- enzyme_pfams_acc
    else:
        pfam_rules_df = pd.DataFrame()

    # CORRELATION
    if Options.correlation_file:
        correlation_df = load_correlations(Options)
    else:
        correlation_df = pd.DataFrame()

    # MERGE
    if Options.gene_annotation and Options.pfam_RR_annotation:
        print_milestone('Integrating annotations, and RR...', Options.verbose)
        merged_df = pd.merge(annotations_df, pfam_rules_df, how='inner')    # on pfam_rule
        del merged_df['pfam_rule']
        # now each rule has suspect genes
    elif Options.pfam_RR_annotation:
        merged_df = pfam_rules_df
        del merged_df['pfam_rule']
    else:
        merged_df = pd.DataFrame()

    if Options.correlation_file:
        print_milestone('Integrating correlations...', Options.verbose)
        merged_df = pd.merge(merged_df, correlation_df, how='inner')    # on gene

    print_milestone('Duplicate cleanup...', Options.verbose)
    merged_df = merged_df.drop_duplicates()         # annotations_df merge produces duplicates due to pfam_rule

    return merged_df
