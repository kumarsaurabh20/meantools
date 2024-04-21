#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm
import warnings
#import janitor
import gizmos

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_args():
    """
    Self-explanatory. Toodles.
    :return:
    """
    parser = argparse.ArgumentParser(prog='queryStructures',
                                     description='Integrate LOTUS database and extract structures based on your mass signatures!',
                                     epilog='Contact - kumar.singh@wur.nl')
    parser.add_argument('-add', '--adducts_file', default=True, action='store', required=True, help='ESI-MS_adducts.csv')
    parser.add_argument('-ms', '--mass_signatures_file', default=True, action='store', required=True, help='Mass signatures two column file: {id,mz}')
    parser.add_argument('-db', '--structures_db', default='lotus', required=True, choices=["lotus", "custom"], help='Use either LOTUS db file or a custom db file!')
    parser.add_argument('-dbp', '--structures_db_path', default=True, action='store', required=True, help='Path of the database file!')
    parser.add_argument('-dtn', '--structures_db_tablename', default=True, action='store', required=True, help='Tablename of the structures db!')
    parser.add_argument('-t', '--taxonomy', default='', required=False, help='Use taxonomy option to filter structures from the lotus database!')
    parser.add_argument('-dn', '--sqlite_db_name', default=True, action='store', required=True, help='Provide a name for the database!')
    parser.add_argument('-tn', '--sqlite_table_name', default=True, action='store', required=True, help='Provide a name for the database table!')
    parser.add_argument('-c', '--chunk_size', action='store', default=10, type=int, required=True, help='Divide the mass signatures file in to chunks!')
    parser.add_argument('-p', '--ppm', action='store', default=20, type=int, required=True, help='Divide the input file into chunks!')
    parser.add_argument('-csv', '--output_csv_files', default=False, action='store_true', required=False, help='Output integration results in to multiple CSV files!')
    parser.add_argument('-o', '--output_folder', default=False, action='store', required=False)
    parser.add_argument('-v', '--verbose', default=False, action='store', required=False)
    return parser.parse_args()


def split_dataframe(df, chunk_size):
    num_chunks = len(df) // chunk_size
    if len(df) % chunk_size != 0:
        num_chunks += 1
    for i in range(num_chunks):
        yield df[i*chunk_size:(i + 1) * chunk_size]


def npdb_to_pd(npdb):
    conn = sqlite3.connect(npdb)
    c = conn.cursor()
    query = c.execute("SELECT structure.structure_id,structure.monoisotopic_mass, structure_has_data_source.source_id, structure_has_data_source.source_name, structure.inchi,structure.inchi_key2,structure.smile,structure.superclass,structure.class,structure.subclass FROM structure left join structure_has_data_source on structure_has_data_source.structure_id = structure.structure_id")
    cols = [column[0] for column in query.description]
    results = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    c.close()

    return results

def custom_db_to_pd(custom, tablename, filtercolumn='', filterText=''):

    keyword = '% filterText %'
    conn = sqlite3.connect(custom)
    c = conn.cursor()
    if filter:
        query = c.execute("SELECT * FROM ? WHERE ? LIKE ?", (tablename, filtercolumn, keyword))
    else:
        query = c.execute("SELECT * FROM ?", (tablename))
    cols = [column[0] for column in query.description]
    results = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    c.close()

    return results

def lotus_to_pd(lotus, taxa):
    
    # Here lotus database is being queried and filtered using specific taxa-related rows

    conn = sqlite3.connect(lotus)
    c = conn.cursor()
    
    if taxa:
        #query = c.execute("SELECT * FROM lotusUniqueNaturalProduct WHERE allTaxa LIKE ?", ('%' + taxa + '%',))
        #query = c.execute(f"SELECT * FROM Jeon_dummy_falcarindiol WHERE allTaxa LIKE ?", ('%' + taxa + '%',))
        query = c.execute(f"SELECT * FROM {Options.structures_db_tablename} WHERE allTaxa LIKE '%{taxa}%'")
    else:
        query = c.execute(f"SELECT * FROM {Options.structures_db_tablename}")

    cols = [column[0] for column in query.description]
    results = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    c.close()

    return results

def checkP(x):
    return x + x*Options.ppm/1000000
def checkN(x):
    return x - x*Options.ppm/1000000

def create_output_files(sqlitedb, sql, columns, filename):
    db_file = os.path.join(sqlitedb)
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cursor = cur.execute(sql)
    count = 1

    signatures = {}
    for each in cursor:
        #print(each)
        #ms_name, mz, mm = each
        signatures[count] = list(each)
        count += 1
    con.close()

    result = pd.DataFrame(signatures.values(), columns=columns)
    result.to_csv(filename, index=False)


def main(Options):

    try:
        # Load the adducts file
        adducts_df = pd.read_csv(Options.adducts_file)
        gizmos.print_milestone('Step 1: Adducts file is loaded!', Options.verbose)
    except FileNotFoundError:
        gizmos.print_milestone("Adducts file is not found. Please check the file or file path.", Options.verbose)
        raise

    try:
        # Load the mass signature file
        df = pd.read_csv(Options.mass_signatures_file, index_col=None, header=0)
        df = df.drop_duplicates(subset=['metabolite'], keep='first')
        gizmos.print_milestone('Step 2: Metabolome file is loaded!', Options.verbose)
    except FileNotFoundError:
        gizmos.print_milestone("Mass signature file is not found. Please check the file or file path.", Options.verbose)
        raise

    # Load the NPDB database
    #db = NPDB_to_pd(Options.structures_sqlite_file)

    # Load the LOTUS database

    if Options.structures_db == "custom":
        db = custom_db_to_pd(Options.structures_db_path, tablename, filtercolumn='', filterText='')
    elif Options.structures_db == "lotus":
        db = lotus_to_pd(Options.structures_db_path, Options.taxonomy)
    else:
        gizmos.print_milestone('Please select an option for structures_db argument!', Options.verbose)
        sys.exit()

    # Create a folder to write all files related with this script.
    if not os.path.exists(Options.output_folder):
        os.makedirs(Options.output_folder)

    # Create an empty sqlite db
    # Kumar Mar 2023
    # script should connect to the database which is previously geenrated by the correlation script
    # Add a new table name to the same database
    # db_file_path = os.path.join(Options.output_folder, Options.sqlite_db_name)
    try:
        conn = sqlite3.connect(Options.sqlite_db_name)
    except sqlite3.Error as e:
        gizmos.print_milestone(f"SQLite connection error: {e}", Options.verbose)
        raise

    #
    df.columns = ['ms_name', 'mz']
    df = df.astype({'mz': float})
    #chunks = split_dataframe(df, Options.chunk_size)
    chunk_count = sum(1 for _ in split_dataframe(df, Options.chunk_size))
    counter = 1
    #gizmos.print_milestone('Step 3: Processing mass signature file...', Options.verbose)
    for each in tqdm(split_dataframe(df, Options.chunk_size), total=chunk_count, desc='Processing m/z chunks', unit='chunk'):
                   
        #gizmos.print_milestone('    Reading chunk {}'.format(counter), Options.verbose)
        chunk_df = pd.DataFrame(each)
        temp = chunk_df.iloc[:, 1]
        ids = chunk_df.iloc[:, 0]

        ## OLD
        ## transform the mz with +/- ppm
        #col1 = temp.transform(checkP)
        #col2 = temp.transform(checkN)

        ## add +/- ppm to the dataframe
        #chunk_df.loc['top'] = col1
        #chunk_df.loc['bottom'] = col2

        # make adducts dataframe comparable to mz i.e add all adducts to the individual feature IDs
        # the expression below is creating a new DataFrame by repeating adducts_df multiple times 
        # (as many times as there are rows in chunk_df) and using ids to label or key the repeated DataFrames 
        # in the resulting concatenated DataFrame. This could be used to repeat a DataFrame's data along 
        # a new axis, effectively expanding it to match the length of another DataFrame (in this case, chunk_df).
        new = pd.concat([adducts_df]*len(chunk_df), keys=ids)
        new.reset_index(inplace=True)

        # merge the chunk_df with all adducts data frame using feature id
        df_outer = pd.merge(chunk_df, new, on="ms_name", how="outer")

        # Kumar Dec 2023
        # low and high masses - these are the mass tolerances for matching the experimental mass
        # first calculate the adduct, and then apply a tolerance to it, 
        # i.e., plus/minus 0.01 Da (or a smaller tolerance if the data is of higher resolution 
        df_outer['mm'] = df_outer["Mult"] * df_outer["mz"] + df_outer["Mass"]
        # transform the mz with +/- ppm
        df_outer['mm_high'] = df_outer["Mult"] * df_outer["mz"] + df_outer["Mass"]
        df_outer['mm_high'] = df_outer['mm_high'].transform(checkP)
        df_outer['mm_low'] = df_outer["Mult"] * df_outer["mz"] - df_outer["Mass"]
        df_outer['mm_low'] = df_outer['mm_low'].transform(checkP)
        
        a = db.molecular_weight.values
        bh = df_outer.mm_high.values
        bl = df_outer.mm_low.values

        # numpy broadcasting (https://stackoverflow.com/questions/44367672/best-way-to-join-merge-by-range-in-pandas)
        # We look for every instance of being greater than or equal to bl
        # while at the same time a is less than or equal to bh.
        i, j = np.where((a[:, None] >= bl) & (a[:, None] <= bh))
        ##
        # we used pandas approach instead of numpy in the generation of final dataframe.
        # This works much faster and consumes less memory.
        # append method linked with concat can get all the NaN values as well from the unmatched rows.
        temp2 = pd.concat([db.loc[i, :].reset_index(drop=True), df_outer.loc[j, :].reset_index(drop=True)], axis=1).append(db[~np.in1d(np.arange(len(db)), np.unique(i))],ignore_index=True, sort=False)

        # drop the NaN rows and get rid of the redundancy by droping duplicates based on inchi key.
        #final = temp2[temp2['mm_high'].notnull()].drop_duplicates(subset=['inchikey'], keep="first")
        #final = temp2[temp2['mm_high'].notnull()].drop_duplicates(subset=['ms_name', 'mz', 'Ion_name'], keep="first")
        #final = temp2[temp2['mm_high'].notnull()].drop_duplicates(subset=['lotus_id', 'smiles', 'ms_name'], keep="first")
        #final = temp2[temp2['mm_high'].notnull()].drop_duplicates(subset=['lotus_id', 'ms_name', 'mz', 'Ion_name'], keep="first")
        final = temp2[temp2['mm_high'].notnull()].drop_duplicates(subset=['lotus_id', 'molecular_weight', 'ms_name', 'mz'], keep="first")

        # Drop unwanted columns from the final dataframe
        final.drop(['level_1', 'Ion_mode', 'Ion_mass', 'Charge', 'Mult', 'Mass', 'mm', 'mm_low', 'mm_high'], axis=1, inplace=True)

        # Push the final dataframe in a SQLite database, with append on as we are looping over chunks.
        #gizmos.print_milestone('    Writing the intersection to the database...', Options.verbose)
        final.to_sql(Options.sqlite_table_name, conn, if_exists = "append" ,index=False)

        counter += 1
    conn.close()

    # Generate NPDB formatted files
    if Options.output_csv_files:
        gizmos.print_milestone('Writing CSV files...', Options.verbose)
        res_file_path = os.path.join(Options.output_folder, 'full_results.csv')
        sql_full = "SELECT lotus_id, ms_name, mz, molecular_weight, iupac_name, inchi, inchikey, smiles, NPclassifierPathway, NPclassifierSuperclass, NPclassifierClass, Ion_name FROM {};".format(Options.sqlite_table_name)
        create_output_files(Options.sqlite_db_name, sql_full, ['lotus_id', 'ms_name', 'mz', 'molecular_weight', 'iupac_name', 'inchi', 'inchikey', 'smiles', 'NPclassifierPathway','NPclassifierSuperclass','NPclassifierClass', 'Ion_name'], res_file_path)

        res_file_path = os.path.join(Options.output_folder, 'LOTUS_entries.csv')
        sql_lotus_entries = "SELECT ms_name, lotus_id, molecular_weight, smiles FROM {};".format(Options.sqlite_table_name)
        create_output_files(Options.sqlite_db_name, sql_lotus_entries, ['ms_name', 'lotus_id', 'molecular_weight', 'smiles'], res_file_path)

        res_file_path = os.path.join(Options.output_folder, 'observed_mm.csv')
        sql_mm = "SELECT ms_name, mz, molecular_weight FROM {};".format(Options.sqlite_table_name)
        create_output_files(Options.sqlite_db_name, sql_mm, ['ms_name', 'mz', 'molecular_weight'], res_file_path)


if __name__ == "__main__":
    Options = get_args()
    start_time = time.time()
    main(Options)
    end_time = time.time()
    print(end_time - start_time)

#def create_connection(db_file):
#    """ create a database connection to a SQLite database """
#    conn = None
#    try:
#        conn = sqlite3.connect(db_file)
#        print(sqlite3.version)
#    except Error as e:
#        print(e)
#    finally:
#        if conn:
#            conn.close()
# if __name__ == '__main__':
#   create_connection(r"C:\sqlite\db\pythonsqlite.db")

#npdb.where(npdb['monoisotopic_mass']>=df_outer['mm_low']).where(npdb['A']<df_outer['mm_high']).dropna()
        #can only compare identically labelled series objects

        #final = npdb.conditional_join(df_outer,('monoisotopic_mass', 'mm_low', '>='),('monoisotopic_mass', 'mm_high', '<='),how = 'left')
        #print(final[final['mm_high'].notnull()])
        #janitor based conditional join statement
        #print(df_outer)

        #SQLlite based solution
        #conn = sqlite3.connect(":memory:")
        #npdb.to_sql("npdb", conn, index=False)
        #df_outer.to_sql("df_outer", conn, index=False)
        #qry = "SELECT * FROM npdb, df_outer WHERE npdb.monoisotopic_mass >= df_outer.mm_low and npdb.monoisotopic_mass <= df_outer.mm_high"
        #tt = pd.read_sql_query(qry,conn)
        #print(tt)

# SQLlite based solution
        # conn = sqlite3.connect(":memory:")
        # npdb.to_sql("npdb", conn, index=False)
        # df_outer.to_sql("df_outer", conn, index=False)
        # qry = "SELECT * FROM npdb, df_outer WHERE npdb.monoisotopic_mass >= df_outer.mm_low and npdb.monoisotopic_mass <= df_outer.mm_high"
        # tt = pd.read_sql_query(qry,conn)
        # print(tt)

        #final.to_csv(filename, index = False)

#print(df_outer)
        #print(df_outer.columns)
        #print(range)

        #df_outer['temp'] = 1
        #npdb['temp'] = 1
        #final = pd.merge(df_outer, npdb, on="temp", how="outer")
        #print(final.columns)

        #conditions = [final['monoisotopic_mass'].ge(final['mm_low']) & final['monoisotopic_mass'].le(final['mm_high'])]
        #choices = [0]
        #final['filter'] = np.select(conditions, choices, default=0)
        #print(final)
        #print(final.columns)

#def split_dataframe(df, chunk_size = 2):
#    chunks = list()
#    num_chunks = math.ceil(len(df)) / chunk_size
#    for i in range(int(num_chunks)):
#        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
#    return chunks
