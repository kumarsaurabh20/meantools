#!/usr/bin/env python
import sys
import subprocess
import os
import argparse
import shutil
import datetime

# import ConfigParser

def get_args():
    parser = argparse.ArgumentParser(description=printStartMessage(), add_help=False)
    optional = parser.add_argument_group('Optional arguments')

    rr = parser.add_argument_group('Reaction Rules database', 'MassTransitions estimate from RetroRules and MetaNetX DB')
    npdb = parser.add_argument_group('NPDB structural database', 'Mass signature file queried to NPDB')
    corr = parser.add_argument_group('Omics correlation', 'Correlation of Metabolomics and Transcriptomics data')
    predict = parser.add_argument_group('Predictions', 'Prediction of pathways')

    optional.add_argument("-h", "--help", action="help", help="show this help message and exit")
    optional.add_argument("-k", "--keep", help="keep all temporary files", action="store_true")
    optional.add_argument('-d', '--decimals', required=False, type=int, default=2, help='Number of decimals kept. Default: 2.')
    # optional.add_argument("-c", "--config", help="Configuration file which specifies commands")
    optional.add_argument('-g', '--ghosts', required=False, default=False, action='store_true', help='Flag. Ghosts mass signatures will be added.')
    optional.add_argument('--verbose', '-v', default=False, action='store_true', required=False)

    rr.add_argument('-rr_db', '--retrorules_sql_file', default=True, action='store_true', required=True, help='mvc.db')
    rr.add_argument('-mnx', '--metanetx_file', default=True, action='store_true', required=True, help='chem_prop.tsv')
    rr.add_argument('-m', '--monoisotopic_mass', default=False, action='store', required=False, help='Flag. Use if metanetx_file has a "mm" column.')
    rr.add_argument('-rr_o', '--rr_output_folder', default=True, action='store', required=True)
    rr.add_argument('-ugr', '--use_greedy_rxn', default=False, action='store_true', required=False, help='Flag. Use to make reaction SMARTS greedier by removing H and valence requirements.')

    npdb.add_argument("-ms", "--mass_signature", default=False, action='store_true', required=True, help='Two-column csv with header. One mass signature per line. Column 1: mass signature ID.' 'Column 2: mz.')
    npdb.add_argument("-npdb_o", "--NPDB_output_folder", required=True)
    npdb.add_argument("-npdb_s", '--npdb_sqlite', default='', required=False, type=str, help='NPDB sqlite file.')
    npdb.add_argument("-mol", "--molecules_table", default='', required=False, action='store_true', help='CSV with molecules to query. Must have header. Format: name,mm,smiles')
    npdb.add_argument("-tol", "--tolerance", default=30, required=False, type=float, help='In ppm. Default: 30.')
    npdb.add_argument("-ion", "--ion_mode", default='', type=str, required=False, choices=['positive', 'negative'], help='Default: both')

    corr.add_argument("-tr", "--transcripts_file", default=False, action='store_true', required=True, help="Transcriptomics quantitation matrix")
    corr.add_argument("-mb", "--metabolites_file", default=False, action='store_true', required=True, help="Metabolomics file with feature IDs and intensities of all samples")
    corr.add_argument("-md", "--mad_filter", default=False, required=False, help='Removes arrays with a MAD of 0.')
    corr.add_argument("-z", "--remove_zeros", default=False, required=False, action='store_true', help='Removes arrays with at least one 0.')
    corr.add_argument("-cm", "--method", default='spearman', type=str, required=False, choices=['spearman', 'pearson', 'pearsonlog'], help='Default is the spearman correlation method; pearsonlog uses log-transformed arrays, where arrays with atleast a zero are always removed first regardless of otheroptions.')
    corr.add_argument("-a", "--annotation", default='', action='store_true', required=False, help='Comma-delimited two-columns file with annotations. No header.')
    corr.add_argument("-p", "--plot", default=False, required=False, action='store_true', help='Plots showing the correlation will be created.')
    corr.add_argument("-cc", "--corr_cutoff", default=0.7, required=False, type=float, help='Minimum absolute correlation coefficient. Default: 0.7. Use 0 for no cutoff.')
    corr.add_argument("-cpc", "--corr_p_cutoff", default=0.1, required=False, type=float, help='Maximum P value of correlation. Default: 0.1. Use 1 for no cutoff.')

    predict.add_argument('-i', '--iterations', '-i', default=5, type=int, required=False, help='Number of iterations. Default: 5')
    predict.add_argument('-qs', '--only_query_small', default=False, action='store_true', required=False, help='Use if we should query only small_rules.')
    predict.add_argument('-mmtd', '--max_mass_transition_diff', default=0.05, type=float, required=False, help='Tolerance for the difference in expected and observed mass_transitions. Default = 0.05')
    predict.add_argument('-usmf', '--use_substrate_mm_in_file', default=False, action='store_true', required=False, help='Flag. Otherwise, mm is recalculated.')
    predict.add_argument('-pfam', '--pfam_RR_annotation_file', default=True, action='store', required=True, help='Nine-column csv. reaction_id, uniprot_id, Pfams, KO, rhea_id_reaction, kegg_id_reaction, rhea_confirmation, kegg_confirmation, KO_prediction')
    predict.add_argument('-gannot', '--gene_annotation_file', default=True, action='store', required=True, help='Two-column csv. Gene, pfam1;pfam2')
    predict.add_argument('-pfamd', '--pfam_RR_annotation_dataset', default='strict', required=False, choices=['strict', 'medium', 'loose'], help='Default: strict.')

    return parser.parse_args()


def checkArguments():

    global args

    if args['rr_db'] == None:
        print('You need to format the databases file first!')
        exit()
    if args['rr_db'] != None and args['mnx'] == None:
        print('You should also provide MetaNetX file in a .csv format along with Retro Rules db.')
        exit()

def checkModules():

    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    modules = ['numpy', 'pandas', 'rdkit', 'scipy', 'matplotlib', 'seaborn', 'networkx']
    for each in modules:
        if each in installed_packages:
            pass
        else:
            print("{} module is not found in the environment!".format(each))

def checkFormatDatabases():

    global db_path
    base_dir_path = os.path.dirname(os.path.realpath(__file__))
    db_path = base_dir_path + "/format_database"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    else:
        if not os.listdir(db_path):
            pass
        else:
            print("Format Database directory is not empty!")

def printStartMessage():
    print(
    """                       
     __  __ _____    _    _   _ _              _     
    |  \/  | ____|  / \  | \ | | |_ ___   ___ | |___ 
    | |\/| |  _|   / _ \ |  \| | __/ _ \ / _ \| / __|
    | |  | | |___ / ___ \| |\  | || (_) | (_) | \__ |
    |_|  |_|_____/_/   \_\_| \_|\__\___/ \___/|_|___/
    """
    )
    print("")
    print("MEANtools::Metabolite Anticipation tools")
    print("Developer/Maintainer: Dr. Kumar Saurabh Singh")
    print("Email: kumar.singh@wur.nl / k.s.singh@uu.nl")
    print("Group Leader: Dr. Marnix H. Medema")
    print("Email: marnix.medema@wur.nl")
    print("Repository: https://git.wageningenur.nl/medema-group/meantools")
    print("")


def convertTimeDeltaToReadableString(timeDelta):
    seconds = timeDelta.seconds
    hours = timeDelta.days * 24
    hours += seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    seconds += timeDelta.microseconds / 1000000.0
    secondString = "{:.1f}".format(seconds)

    returnString = ""
    if hours > 0:
        return str(hours) + ' h, ' + str(minutes) + ' min, ' + secondString + ' s'
    if minutes > 0:
        return str(minutes) + ' min, ' + secondString + ' s'
    return secondString + ' s'


def printEndMessage():
    print("############################################")
    print("MEANtools::Pathway prediction is finished now!")
    print("Total time to complete: %d" % convertTimeDeltaToReadableString())
    print("############################################")


def getDateTimeString():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    startTime = datetime.datetime.now()
    global args
    args = vars(get_args())

    return


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':

    try:
        checkModules()
    except:
        print("Install the missing module(s)!")

    try:
        checkFormatDatabases()
    except:
        print("Something happend while searching for database folder!")

    main()
