import os
import sys
import re
import glob
import argparse
#
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import markov_clustering as mc
import networkx as nx
import sqlite3
#
import gizmos

#
plt.rcParams['figure.max_open_warning'] = 200
#

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-ft','--feature_table', default='', action='store', required=True, help='Metabolomics-based feature table!')
	parser.add_argument('-qm','--quantitation_matrix', default='', action='store', required=True, help='Normalized expression table from the RNAseq data!')
	parser.add_argument('-l','--targeted_list', default='', action='store', required=False, help='Target list of genes and metabolites in a csv file. <ID><metabolites><genes>')
	parser.add_argument('-a', '--annotation', default=False, action='store_true', required=False, help='Flag for annotation. Default is False')
	parser.add_argument('-f', '--heatmap', default=False, action='store_true', required=False, help='Flag for generating heatmaps for each FCs. Default is False')
	parser.add_argument('-g','--annotation_file', default='', action='store', required=False, help='Annotation file to annotate FCs. Only if -a is flagged! <gene><tab><annotation>')
	parser.add_argument('-mc', '--merge_clusters', default=False, action='store_true', required=False, help='If you want to merge FCs!')
	parser.add_argument('-mm', '--merge_method', default='overlap', required=False, choices=['fingerprinting', 'overlap'], help='Default: overlap!')
	parser.add_argument('-dr', '--decay_rate', default=25, type=int, required=False, help='Deacy rate of FCs. Default: 25')
	parser.add_argument('-e', '--evidence', default=False, action='store_true', required=False, help='Flag. Additional evidences like new edge scores from co-expression or spectral networking!')
	parser.add_argument('-es', '--evidence_source', default='', required=False, choices=['coex', 'specnet'], help='Default: edge scores from co-expressio network. Other option is edge scores from spectral networking using metabolomics MS/MS data!')
	parser.add_argument('-o', '--outfile', default=False, required=False, action='store', help='Name of the matrix file showing association strength (WARNING:: file size is dependent on number of FCs)!')
	parser.add_argument('-dn', '--sqlite_db_name', default='', action='store', required=True, help='Provide a name for the database')
	return parser.parse_args()


def association_strength(Fi, Fj):
	# this formula is adapted from the study https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-1-54
	# to capture the strenth of association between two decomposed singular vectors which in our case are called fingerprints. 
	# In the present work we try to capture the strength of fingerprints between clusters
	corr_coeff = np.corrcoef(Fi, Fj)[0, 1]
	strength = (1 + corr_coeff) / 2
	return strength


def calculate_sigma(df):
	# Calculate the singular value decomposition (SVD) for the DataFrame 'df'
	# Here, I'm using 'Column1' and 'Column2' as the columns for the SVD calculation
	# Replace them with your actual column names
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	X = df[numeric_cols].values
	U, s, VT = np.linalg.svd(X, full_matrices=False)
	#sigma = np.diag(s) ## use this if you need a matrix instead of a 1D vector
	sigma = VT[0]	## VERY IMPORTANT::First right singular vector
	return sigma


def annotate_clusters(row, gene_col_index, output_folder):

	# parsing annotation file
	annotation_df = pd.read_csv(Options.annotation_file)
	cluster_id = row[0]
	elements_str = row[gene_col_index]
	elements_list = elements_str.split()
	extracted_rows_combined = pd.DataFrame()

	for element in elements_list:
		if element in annotation_df['gene'].values:
			extracted_row = annotation_df[annotation_df['gene'] == element]
			extracted_rows_combined = pd.concat([extracted_rows_combined, extracted_row], ignore_index=True)
		else:
			pass

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	filename = cluster_id + ".csv"
	output_file = os.path.join(output_folder, filename)
	extracted_rows_combined.to_csv(output_file, index=False)

	return

def generate_heatmaps(df, cluster_id, directory):

	df.set_index('feature', inplace=True)

	# Create the directory for the heatmaps, if it doesn't exist
	if not os.path.exists(directory):
		os.makedirs(directory)
  	
	# Calculate the size based on the number of features
	#num_features = len(extracted_rows_combined.columns)
	#figsize = (8, num_features // 2)  # Adjust this factor to control the height

	# Determine the heatmap height based on the number of rows
	# Adjust the factor (0.3) as needed
	heatmap_height = len(df) * 0.3

	# Create the heatmap with dynamic height
	# Adjust the figsize width (10) as needed
	plt.figure(figsize=(10, heatmap_height))

	# Generate the heatmap with the calculated figsize
	sns.heatmap(df, annot=False, cmap='coolwarm', linewidths=0.5)

	# Save the heatmap as PNG in the specified folder
	filename = os.path.join(directory, '{}.png'.format(cluster_id))
	plt.savefig(filename, bbox_inches='tight')

	return


def split_and_match(row, metabolite_df, column):

	'''
	The function splits elements and return matching/unmatching
	Splits genes and metabolites into separate columns

	:param1 row: row of the dataframe
	:param2 metabolite_df: because it checks metabolic features within a merged_cluster, unmatched elements automatically assigned to genes column
	
	:return : returns two combined series which can be integrated into a data frame
	'''

	merged_members = row[column].split(',')
	matching = [metabolite for metabolite in merged_members if metabolite in metabolite_df['feature'].to_list()]
	unmatching = [metabolite for metabolite in merged_members if metabolite not in matching]
	return pd.Series([matching, unmatching])

def merge_clusters_overlapping(df, metabolite_features, dr=25):
	
	'''
	This method uses the networkx library to represent the relationships between genes and metabolites as a graph. 
	It then finds connected components in the graph, which correspond to clusters that share genes and metabolites
	
	:param frame: Collective data frame from all DR representing a combined collection of FCs. <ID, Members>
	:return : returns a data frame with merged clustes <new_ID+merged_clusters, combined_members>

	'''

	# Selecting rows using a user provided decay rate
	# Logic: FCs will definitly overlap when you combine all DRs, 
	# but there might be non-overlapping FCs within a single DR
	decay_rate = "DR_" + str(dr)
	selected_rows = df[df['Source'] == decay_rate]
	df = selected_rows[['ID', 'Members']]
	#df[['metabolites', 'genes']] = df.apply(split_and_match, args=(metabolite_df, 'Members'), axis=1)

	# Split the 'components' column into a list of genes and metabolites
	# df.loc[:, 'Members_list'] = df.loc[:, 'Members'].apply(lambda x: x.split())
	# Trying to avoid SettingWithCopyWarning by creating a new df variable
	new_df = df.assign(Members_list=df['Members'].apply(lambda x: x.split()))

	merged_content = []
	

	for metabolite in metabolite_features:
		for index, row in new_df.iterrows():
			if metabolite in row['Members_list']:
				merged_content.extend(row['Members_list'])
			else:
				pass

	
	merged_content = list(merged_content)
	print(merged_content)


	# Create a graph to represent connections between genes and metabolites
	G = nx.Graph()

	# Add edges between genes and metabolites within each cluster
	for _, row in tqdm(new_df.iterrows(), total=new_df.shape[0], desc='Checking overlap between FCs'):
		cluster_nodes = row['Members_list']
		#G.add_nodes_from(cluster_nodes)
		#G.add_edges_from([(node, cluster_metabolites[-1]) for node in cluster_metabolites[:-1]])

		# Extract metabolites and genes from cluster_nodes
		cluster_metabolites = [node for node in cluster_nodes if node in metabolite_features]
		cluster_genes = [node for node in cluster_nodes if node not in metabolite_features]

		# Add metabolites as nodes to the graph
		G.add_nodes_from(cluster_metabolites)
		# Add edges between metabolites
		G.add_edges_from([(metabolite1, metabolite2) for metabolite1 in cluster_metabolites for metabolite2 in cluster_metabolites if metabolite1 != metabolite2])

		

	# Find connected components (clusters) in the graph
	connected_components = list(nx.connected_components(G))

	# Create a mapping from gene/metabolite to cluster ID
	node_cluster_mapping = {node: cluster_id for cluster_id, nodes in enumerate(connected_components, start=1) for node in nodes if node in metabolite_features}
	#metabolite_cluster_mapping = {metabolite: cluster_id for cluster_id, metabolites in enumerate(connected_components, start=1) for metabolite in metabolites}

	# Display the merged clusters with original cluster IDs
	merged_clusters = {}
	for i, cluster in enumerate(connected_components, start=1):
		original_cluster_ids = {int(node_cluster_mapping[node]) for node in cluster}
		merged_cluster_id = f'MC{i}'
		merged_clusters[merged_cluster_id] = cluster

	# Retrieve genes associated with each merged cluster
	merged_genes = {}
	for merged_cluster_id, original_cluster_ids in merged_clusters.items():
		genes_in_merged_cluster = []
		for original_cluster_id in original_cluster_ids:

			# Retrieve genes from the original cluster
			genes_in_original_cluster = [node for node in connected_components[original_cluster_id] if node not in metabolite_features]
			genes_in_merged_cluster.extend(genes_in_original_cluster)

		merged_genes[merged_cluster_id] = genes_in_merged_cluster


	metabolites_data = []
	for merged_cluster_id, nodes in merged_clusters.items():
		metabolites_in_cluster = [node for node in nodes if node in metabolite_features]
		metabolites_data.append({'ID': merged_cluster_id, 'Metabolites': ', '.join(metabolites_in_cluster)})

	metabolites_df = pd.DataFrame(metabolites_data)
	print(metabolites_df)

	genes_data = []
	for merged_cluster_id, nodes in merged_clusters.items():
		genes_in_cluster = [node for node in nodes if node not in metabolite_features]
		genes_data.append({'ID': merged_cluster_id, 'Genes': ', '.join(genes_in_cluster)})

	genes_df = pd.DataFrame(genes_data)
	print(genes_df)

	
	merged_df = pd.merge(metabolites_df, genes_df, on='ID')

	#merged_list = [{'ID': key, 'Merged Members': ','.join(value)} for key, value in merged_clusters.items()]
	#merged_df = pd.DataFrame(merged_list)

	#merged_df[['metabolites', 'genes']] = merged_df.apply(split_and_match, args=(metabolite_df, 'Merged Members'), axis=1)
	#columns_to_keep = ['ID', 'metabolites', 'genes']
	#merged_df = merged_df.loc[:, columns_to_keep]
	#merged_df['metabolites'] = merged_df['metabolites'].apply(lambda x: ', '.join(x))
	#merged_df['genes'] = merged_df['genes'].apply(lambda x: ', '.join(x))

	return merged_df

def merge_clusters_fingerprinting(raw_df, corr_df, metabolite_df, threshold=0.5, inflation=2.0):

	'''
	Pseudopipeline
	---------------

	1. Load the raw (cluster_id\tMembers) & correlation (correlation matrix) DataFrame.
	2. Initialize a list to keep track of which clusters have been merged.
	3. Loop through the DataFrame to find clusters that meet the merging criteria.
	4. Merge clusters based on the criteria and store the merged cluster information in a new DataFrame.
	5. Update the list of merged clusters to ensure they are not considered for further merging.
	6. Continue looping until no more merging is possible.
	7. Save the merged DataFrame to a new file.
	'''

	# Create a graph using NetworkX
	G = nx.Graph()

	# Add nodes and edges based on the correlation matrix
	for i, row in tqdm(corr_df.iterrows(), total=corr_df.shape[0], desc='Merging FCs'):
		G.add_node(i)
		for j, value in row.items():
			if i != j and value >= threshold:  # Adjust the threshold as needed
				G.add_edge(i, j, weight=value)

	# Convert the NetworkX graph to a SciPy sparse matrix
	adjacency_matrix = nx.to_scipy_sparse_array(G)

	# Apply the MCL algorithm
	result = mc.run_mcl(adjacency_matrix)

	# Extract clusters from the result
	clusters = mc.get_clusters(result)

	# Retrieve column names corresponding to cluster values
	clustered_columns = [[corr_df.index[i] for i in cluster] for cluster in clusters]

	# Print the clusters with column names
	#for i, cluster in enumerate(clustered_columns, 1):
	#	print(f'Cluster {i}: {cluster}')

	#Return a dataframe with cluster ID and its corresponding genes and metabolites
	cluster_df = pd.DataFrame([','.join(cluster) for cluster in clustered_columns], columns=['Cluster'])
	
	cluster_df.columns = ['Cluster']

	# Split the 'Cluster' column in the first DataFrame into separate IDs
	cluster_df['Cluster'] = cluster_df['Cluster'].str.split(',')

	# Create a mapping dictionary for IDs and Members
	id_to_members = raw_df.set_index('ID')['Members'].to_dict()

	# Merge the 'Members' based on IDs in the first DataFrame
	def merge_members(cluster):
		members = [id_to_members.get(cluster_id, '') for cluster_id in cluster]
		return ', '.join(members)

	cluster_df['Merged Members'] = cluster_df['Cluster'].apply(merge_members)

	cluster_df[['metabolites', 'genes']] = cluster_df.apply(split_and_match, args=(metabolite_df,), axis=1)

	# Convert the list in the 'Cluster' column to a string
	cluster_df['Cluster'] = cluster_df['Cluster'].apply(lambda x: ', '.join(x))
	cluster_df['metabolites'] = cluster_df['metabolites'].apply(lambda x: ', '.join(x))
	cluster_df['genes'] = cluster_df['genes'].apply(lambda x: ', '.join(x))

	# drop the merged columns
	cluster_df = cluster_df.drop(columns=['Merged Members'])

	return cluster_df

	'''
	To investigate the optimal inflation value and modularity
	----------------------------------------------------------

	for inflation in [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6]:
	result = mc.run_mcl(adjacency_matrix, inflation=inflation)
	clusters = mc.get_clusters(result)
	Q = mc.modularity(matrix=result, clusters=clusters)
	print("inflation:", inflation, "modularity:", Q)

	'''


def calc_metafingerprints(row, transcripts_df, metabolite_df):
	
	# Create an empty dataframe to store expression and abundance values in genes and metabolites
	extracted_rows_combined = pd.DataFrame()

	cluster_id = row[0]
	elements_str = row[1]
	elements_list = elements_str.split()

	
	for element in elements_list:
		if element in transcripts_df['feature'].values:
			extracted_row = transcripts_df[transcripts_df['feature'] == element]
			extracted_rows_combined = pd.concat([extracted_rows_combined, extracted_row], ignore_index=True)
		elif element in metabolite_df['feature'].values:
			extracted_row = metabolite_df[metabolite_df['feature'] == element]
			extracted_rows_combined = pd.concat([extracted_rows_combined, extracted_row], ignore_index=True)
		else:
			sys.exit("element is not found in either data frame!")

	# save all the heatmaps for FCs in a folder named 'heatmaps'
	if Options.heatmap:
		generate_heatmaps(extracted_rows_combined, cluster_id, "FC_heatmaps")

	# Return the right singular matrix by decomposing a matrix using svd.
	sigma = calculate_sigma(extracted_rows_combined)

	row['metafingerprints'] = sigma
	return row


def main(Options):

	# Raw expression and features file.
	# Used for computing fingerprints
	transcripts = pd.read_csv(Options.quantitation_matrix)
	transcripts = transcripts.rename(columns={'gene': 'feature'})
	metabolites = pd.read_csv(Options.feature_table)
	metabolites = metabolites.rename(columns={'metabolite': 'feature'})

	# Keep common labels between the two datasets
	# It is important to keep the gene and metabolite column names (from the main omics dataset) 
	# as 'feature'
	transcripts_labels = set(transcripts.columns)
	metabolites_labels = set(metabolites.columns)
	common_labels = sorted(list(transcripts_labels.intersection(metabolites_labels)))
	transcripts_df = transcripts[common_labels]
	metabolite_df = metabolites[common_labels]
	
	# scale the transcriptomics and metabolomics datasets by using z-scores 
	# Especially for heatmaps
	t_numeric_columns = transcripts_df.copy()
	t_numeric_columns[t_numeric_columns.select_dtypes(['float64', 'int64']).columns] = t_numeric_columns.select_dtypes(['float64', 'int64']).apply(zscore)
	transcripts_df = t_numeric_columns
	#
	m_numeric_columns = metabolite_df.copy()
	m_numeric_columns[m_numeric_columns.select_dtypes(['float64', 'int64']).columns] = m_numeric_columns.select_dtypes(['float64', 'int64']).apply(zscore)
	metabolite_df = m_numeric_columns

	# Read the cluster files with multiple DR
	all_files = gizmos.import_from_sql(Options.sqlite_db_name, sqlite_tablename="", df_columns=[], conditions={}, structures = False, clone = True)
	
	if not all_files:
		raise ValueError(f"No data was imported from {Options.sqlite_db_name}.")

	# Adding the DR info to know which cluster is coming from which file
	# DR is deacy rate used to calculate clusters
	for table_name, df in all_files.items():
		match = re.search(r'(DR_\d+)', table_name)
		if match:
			source_value = match.group(1)
			df['Source'] = source_value

	# Concatenate all DataFrames into a single DataFrame
	frame = pd.concat(all_files.values(), axis=0, ignore_index=True)

	list_of_counts = []
	for i in range(1, frame.shape[0] + 1):
		list_of_counts.append(f'cluster{i:03d}')
	frame['cluster_id'] = list_of_counts

	# merge cluster ID and source info to create a new ID
	#frame["ID"] = frame["cluster_id"].str.cat(frame["Source"], sep = "_")
	frame["ID"] = frame["cluster_id"] +"_"+ frame["Source"] + "_" + frame["Cluster"].astype(str)

	clusters_frame = frame[['ID', 'Members']]

	for idx, row in tqdm(clusters_frame.iterrows(), total=clusters_frame.shape[0], desc='Annotating FCs'):
		if Options.annotation:
			annotate_clusters(row, 1, 'FC_cluster_annotation')
	
	if Options.merge_clusters:

		if Options.merge_method == "overlap":

			if Options.targeted_list:

				targeted_metabolite_df = pd.read_csv(Options.targeted_list)
				targeted_metabolite_df['metabolites'] = targeted_metabolite_df['metabolites'].str.split(', ')
				df_exploded = targeted_metabolite_df.explode('metabolites')
				combined_metabolites = df_exploded['metabolites'].tolist()
				deduplicated_metabolites = list(set(combined_metabolites))

				merged_df = merge_clusters_overlapping(frame, deduplicated_metabolites, Options.decay_rate)

			else:
				
				metabolite_features = metabolite_df['feature'].to_list()
				merged_df = merge_clusters_overlapping(frame, metabolite_features, Options.decay_rate)

			# send results to the database
			tablename = "merged_cluster_overlap_metabolites"
			
			if Options.decay_rate:
				
				table_name = tablename + "_DR_" + str(Options.decay_rate)
				#gizmos.export_to_sql(Options.sqlite_db_name, merged_df, table_name, index=False)

			else:
				
				pass
				#gizmos.export_to_sql(Options.sqlite_db_name, merged_df, tablename, index=False)
			
		elif Options.merge_clusters == "fingerprinting":

			clusters_with_fingerprints_df = pd.DataFrame()
			results_rows = []

			# Calculate representative fingerprint for each cluster
			for idx, row in tqdm(clusters_frame.iterrows(), total=clusters_frame.shape[0], desc='Processing FCs'):
			
				results_rows.append(calc_metafingerprints(row, transcripts_df, metabolite_df))
			clusters_with_fingerprints_df = pd.concat(results_rows, axis=1).T
			#clusters_with_fingerprints_df = clusters_frame.apply(calc_metafingerprints, axis=1, result_type='expand')


			correlation_matrix = np.zeros((clusters_with_fingerprints_df.shape[0], clusters_with_fingerprints_df.shape[0]))

			# Calculate corrleation strength between two fingerprints that reresent 
			# genes co-expression & metabolites co-abundance profiles for each cluster
			for i in tqdm(range(clusters_with_fingerprints_df.shape[0]), desc="Computing association between FCs"):
				for j in range(clusters_with_fingerprints_df.shape[0]):
					if i == j:
						correlation_matrix[i, j] = 1.0
					else:
						Fi = clusters_with_fingerprints_df.loc[i, 'metafingerprints']
						Fj = clusters_with_fingerprints_df.loc[j, 'metafingerprints']
						correlation_matrix[i, j] = association_strength(Fi, Fj)

			correlation_df = pd.DataFrame(correlation_matrix, columns=clusters_with_fingerprints_df['ID'], index=clusters_with_fingerprints_df['ID'])
		
		
			## Kumar (Nov 2023)
			## The file size explodes at this step.
			## Also the number of columns in SQLite has a limitation of 14k columns.
			#correlation_df.to_csv(Options.outfile)

			cluster_dataframe = merge_clusters_fingerprinting(clusters_frame, correlation_df, metabolite_df, 0.5, 2.0)

			# send results to the database
			gizmos.export_to_sql(Options.sqlite_db_name, cluster_dataframe, 'merged_clusters_fingerprint', index=False)

		else:
			sys.exit("Not a valid method for merging cluster. Try again!")


if __name__ == "__main__":

	Options = get_args()
	main(Options)
