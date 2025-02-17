#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
from os.path import isfile
import sys
import glob
import argparse
import pandas as pd
import subprocess
import numpy as np
import re
#
import json
#
import torch
from torch_geometric.data import Data
from rdkit.Chem import Draw
import requests
#
from IPython.display import SVG
#
from gnn_som import createGnnSom, loadGnnSomState
from gnn_som.MolFromKcf import MolFromKcfFile
#
from io import BytesIO
from rdkit import Chem
from gnn_som.MolFromKcf import MolFromKcfContents
#
import xml.etree.ElementTree as ET


def predict_SOM_labels(enzyme="", smile=""):

	#load config file
	with open('models/config_ori.json', 'r') as f:
		config = json.load(f)
	config['features']['enzyme'] = [tuple(ec) for ec in config['features']['enzyme']] 

	#configure several variants of the model, each was trained on a different data split and all will contribute to the final SOM prediction
	models = []
	for i, params in enumerate(config['models']):
		model = createGnnSom(*config['models'][i])
		loadGnnSomState(model, torch.load('models/model%d.pt' % i, map_location=torch.device('cpu')))
		models.append(model)

	temp = Chem.MolFromSmiles(smile)
	temp_molFile = BytesIO(Chem.MolToMolBlock(temp).encode('ascii'))
	kcf = requests.post('http://rest.genome.jp/mol2kcf/', files={'molfile': temp_molFile}).content.decode('ascii')
	mol = MolFromKcfContents(kcf)

	#construct a graph representation of the molecule, which includes atom feature vectors and bond connectivity.
	numFeatures = sum(len(feature) for feature in config['features'].values())
	x = torch.zeros((mol.GetNumAtoms(), numFeatures), dtype=torch.float32)
	for atom in mol.GetAtoms():
		x[atom.GetIdx(), config['features']['enzyme'].index(enzyme)] = 1
		offset = len(config['features']['enzyme'])
		x[atom.GetIdx(), offset + config['features']['element'].index(atom.GetSymbol())] = 1
		offset += len(config['features']['element'])
		x[atom.GetIdx(), offset + config['features']['kcfType'].index(atom.GetProp('kcfType'))] = 1

	edgeIndex = torch.zeros((2, mol.GetNumBonds() * 2), dtype=torch.int64)
	for bond in mol.GetBonds():
		i = bond.GetIdx()
		edgeIndex[0][i * 2] = bond.GetBeginAtomIdx()
		edgeIndex[1][i * 2] = bond.GetEndAtomIdx()
		edgeIndex[0][i * 2 + 1] = bond.GetEndAtomIdx()
		edgeIndex[1][i * 2 + 1] = bond.GetBeginAtomIdx()

	data = Data(x=x, edgeIndex=edgeIndex)

	#predict SOM labels using multiple models and retain the average value for each atom.
	y = None
	for model in models:
		newY = torch.sigmoid(model(data.x, data.edgeIndex))
		y = newY if y is None else torch.add(y, newY)
	y = torch.div(y, len(models))
	
	return y


def extract_reaction_details():
	for annotation in root.findall('.//ANNOTATION'):
		for fingerprints in annotation.findall('FINGERPRINTS'):
			if fingerprints.get('BC'):
				formed_cleaved = fingerprints.find('FORMED_CLEAVED').text
				print("Bond Cleavage:", formed_cleaved)
			if fingerprints.get('RC'):
				centres = fingerprints.find('CENTRE').text
				print("Reaction Centres:", centres)


def get_atom_info(mol):
	"""Extracts atom information from a molecule."""
	info = {}
	for atom in mol.GetAtoms():
		info[atom.GetAtomMapNum()] = (atom.GetSymbol(), atom.GetIsotope())
	return info

def check_elements_in_substructures(substructures, required_elements):
	
	valid_substructures = []
	if 'H' in required_elements:
		required_elements.remove('H')
	
	for substructure in substructures:
		mol = Chem.MolFromSmiles(substructure)
		
		if mol is None:
			continue  #skip invalid SMILES strings
		
		present_elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
		
		if required_elements.issubset(present_elements):
			valid_substructures.append(substructure)
	
	return valid_substructures


def convert_to_set(tup):
	resulting_set = set()  #initialize an empty set

	#iterate over each item in the outer tuple
	for sub_tup in tup:
		#check if the item is a tuple and convert to set
		if isinstance(sub_tup, tuple):
			resulting_set.update(sub_tup)
		else:
			#directly add the item if it's not a tuple
			resulting_set.add(sub_tup)

	return resulting_set


def extract_and_analyze_substructures(xml_file_path):

	#Load XML and parse it
	tree = ET.parse(xml_file_path)
	root = tree.getroot()

	aam_text = root.find('.//MAPPING/AAM').text
	centre_text = root.find('.//FINGERPRINTS/CENTRE').text
	centre_text = centre_text[1:-1]
	#formed_cleaved_text = root.find('.//FINGERPRINTS/FORMED_CLEAVED').text
	fingerprints = root.find(".//FINGERPRINTS[@BC='1']")

	# tags under FINGERPRINTS[@BC='1'] is not fixed
	formed_cleaved_text = ""

	if fingerprints is not None:
		#we expect to find at least one child element under this <FINGERPRINTS> tag
		for child in fingerprints:
			formed_cleaved_text = child.text
			#print(f"Tag: {child.tag}, Text: {child.text}")
	else:
		print("No <FINGERPRINTS BC='1'> tag found.")
	
	# Find all chemical symbols in FORMED_CLEAVED
	formed_cleaved_symbols = set(re.findall(r'[A-Z][a-z]?', formed_cleaved_text))

	# Split the AAM text for substrate and product
	substrate_smiles, product_smiles = aam_text.split('>>')

	# Process the CENTRE tag to extract relevant substructures
	substructures = []
	singletons = set(['[F]', '[O]', '[OH]', '[Cl]', '[N]', '[Br]', '[I]', '[S]', '[C]', '[CH]', '[H]'])
	#centre_text = centre_text.strip('[]')  # Remove surrounding brackets if present

	for part in centre_text.split(', '):
		part_string = re.findall(r'[A-Z][a-z]?', part)
		if '>>' in part:
			substructure = part.split('>>')[0].split(':')[0]
		else:
			substructure = part.split(':')[0]

		if '1,0' in part and substructure not in singletons: # and formed_cleaved_symbols.issubset(set(part_string)):
			substructures.append(substructure)

	valid_substructures = check_elements_in_substructures(substructures, formed_cleaved_symbols)

	#print(f'Valid substructures :: {valid_substructures}')

	# for choosing longest substructure
	#longest_substructure = ""
	#max_length = 0
	#for element in valid_substructures:
	#	cleaned_element = re.sub(r'[^A-Za-z]', '', element)
	#	if len(cleaned_element) > max_length:
	#		max_length = len(cleaned_element)
	#		shortest_substructure = element
	#substructure_molecule = Chem.MolFromSmarts(longest_substructure)
	#matches = substrate_mol.GetSubstructMatches(substructure_molecule)

	
	#sort substructures by length of their cleaned version
	sorted_substructures = sorted(valid_substructures, key=lambda x: len(re.sub(r'[^A-Za-z]', '', x)))
	substrate_mol = Chem.MolFromSmiles(substrate_smiles)

	#print(f'Sorted substructures :: {sorted_substructures}')


    #iterate over sorted substructures and find the first match
	for substructure in sorted_substructures:
		substructure_molecule = Chem.MolFromSmarts(substructure)
		matches = substrate_mol.GetSubstructMatches(substructure_molecule)
		if matches:  #check if there are matches
			#print(f"Matching substructure: {substructure} with matches: {matches}")
			matches = convert_to_set(matches)
			return matches

	#RDKit molecule objects from the SMILES
	#substrate_mol = Chem.MolFromSmiles(substrate_smiles)
	#product_mol = Chem.MolFromSmiles(product_smiles)

	#list_of_matches = []

	#apply each substructure to the main molecule (example with substrate)
	#for substructure_smiles in valid_substructures:
	#	substructure_molecule = Chem.MolFromSmarts(substructure_smiles)
	#	matches = substrate_mol.GetSubstructMatches(substructure_molecule)
	#	list_of_matches.append(matches)

	#unique_matches = set()
	#for item in list_of_matches:
	#	if item:
	#		for subitem in item:
	#			unique_matches.update(subitem)

	#return unique_matches
	#return matches


def compare_molecules(substrate_smiles, product_smiles, enzyme_tuple, unique_matches):
	"""Compares two molecules and identifies changed atoms, calculates likelihoods."""

	#substrate_mol = Chem.MolFromSmiles(substrate_smile_mapped)
	#product_mol = Chem.MolFromSmiles(product_smile_mapped)
	substrate_y = predict_SOM_labels(enzyme_tuple, substrate_smiles)
	#print("#### Substrate SOM")
	#print(substrate_y)
	product_y = predict_SOM_labels(enzyme_tuple, product_smiles)
	#print("#### Product SOM")
	#print(product_y)
	#substrate_info = get_atom_info(substrate_mol)
	#print(substrate_info)
	#product_info = get_atom_info(product_mol)
	#print(product_info)

	substrate_likelihood = []
	product_likelihood = []

	
	if unique_matches:
		for atom_num in unique_matches:
			#print(f"atom_num: {atom_num}, substrate_y size: {substrate_y.size}")
			substrate_likelihood.append(substrate_y[atom_num - 1].item())
		max_substrate_likelihood = max(substrate_likelihood) if substrate_likelihood else 0
		#average_substrate_likelihood = sum(substrate_likelihood) / len(substrate_likelihood) if substrate_likelihood else 0
		max_product_likelihood = max(product_likelihood) if product_likelihood else 0
	else:
		max_substrate_likelihood = 0.0
		max_product_likelihood = 0.0
	
	
	#reaction_likelihood = max_substrate_likelihood * max_product_likelihood

	#return substrate_y, product_y, average_substrate_likelihood #max_substrate_likelihood #, max_product_likelihood, reaction_likelihood
	return substrate_y, product_y, max_substrate_likelihood, max_product_likelihood #, reaction_likelihood


def parse_smiles(smile):
    
    smile_details = {}
    
    atom_pattern = r"\[([A-Za-z#0-9:]+):(\d+)\]"
    bond_pattern = r"([=#]?)\[([A-Za-z#:0-9]+):(\d+)\]"
    branches = re.findall(r"\(\[([^\]]+)\]\)", smile)
    
    atoms = re.findall(atom_pattern, smile)
    
    for i, (atom, position) in enumerate(atoms):
        

    	#skip hydrogen atoms (H:<number>) 
    	#out of bounds Error when H is assigned a position in the SMILES by ReactionDecoder
        if atom == "H":
            continue

        neighbors = []
        
        #slice the SMILES string for preceding and succeeding bonds
        current_index = smile.find(f"[{atom}:{position}]")
        preceding_slice = smile[:current_index]
        succeeding_slice = smile[current_index + len(f"[{atom}:{position}]"):]
        
        if i > 0:
            prev_atom, prev_position = atoms[i - 1]
            if prev_atom != "H":  #skip hydrogens in neighbors
            	preceding_bond_match = re.search(bond_pattern, preceding_slice[::-1])  # Reverse for easier matching
            	preceding_bond = preceding_bond_match.group(1)[::-1] if preceding_bond_match else "-"
            	neighbors.append({
                	"Bond": preceding_bond,
                	"Atom": prev_atom,
                	"Position": prev_position
				})

        if i < len(atoms) - 1:
            next_atom, next_position = atoms[i + 1]
            if next_atom != "H":  #akip hydrogens in neighbors
            	succeeding_bond_match = re.search(bond_pattern, succeeding_slice)
            	succeeding_bond = succeeding_bond_match.group(1) if succeeding_bond_match else "-"
            	neighbors.append({
            		"Bond": succeeding_bond,
                	"Atom": next_atom,
                	"Position": next_position
                })
            
        
        #add branching if defined in parentheses
        #branches = re.findall(rf"\({atom}.*?\)", smile[smile.find(f"[{atom}:{position}]") + len(f"[{atom}:{position}]"):])
        branching = []
        for branch in branches:
            if branch == f'{atom}:{position}':
                    branching.append(branch)
            else:
                branching.append("")
        
        #store the atom details
        smile_details[position] = {
            "Atom": atom,
            "Neighbors": neighbors,
            "Branches": branching
        }
        
    return smile_details
            

    
def compare_smiles_dicts(dict1, dict2):
    
    differences = {}

    #combine all positions from both dictionaries
    all_positions = set(dict1.keys()).union(set(dict2.keys()))

    for position in all_positions:
        atom1 = dict1.get(position, None)
        atom2 = dict2.get(position, None)

        #initialize a dictionary for storing position-specific differences
        position_diff = {}

        if atom1 and atom2:
            
            #compare atom types
            if atom1["Atom"] != atom2["Atom"]:
                position_diff["Atom"] = (atom1["Atom"], atom2["Atom"])

            #compare neighbors (bonds and adjacent atoms)
            neighbors1 = {(neighbor["Bond"], neighbor["Atom"], neighbor["Position"]) for neighbor in atom1["Neighbors"]}
            neighbors2 = {(neighbor["Bond"], neighbor["Atom"], neighbor["Position"]) for neighbor in atom2["Neighbors"]}
            if neighbors1 != neighbors2:
                position_diff["Neighbors"] = {
                    "In Dict1 Only": neighbors1 - neighbors2,
                    "In Dict2 Only": neighbors2 - neighbors1
                }

            #compare branches
            #branches1 = set(atom1.get("Branches", []))
            #branches2 = set(atom2.get("Branches", []))
            #if branches1 != branches2:
            #    position_diff["Branches"] = {
            #        "In Dict1 Only": branches1 - branches2,
            #        "In Dict2 Only": branches2 - branches1
            #    }

        elif atom1:
            #atom missing in dict2
            position_diff["Missing in Dict2"] = atom1

        elif atom2:
            #atom missing in dict1
            position_diff["Missing in Dict1"] = atom2

        if position_diff:
            differences[position] = position_diff

            
    #return positions from the substrate (smiles1) only
    pos_list = [int(position) for position in differences if position in dict1]

    #if you want positions from both substrate and product then uncomment below and comment above
    #pos_list=[]
    
    #print("Differences Between Parsed SMILES Dictionaries:")
    #for position, diff in differences.items():
     #   print(f"Position {position}: {diff}")
        #pos_list.append(position)
    
    
    return pos_list


def extract_and_analyze_SMILES(xml_file_path):

	#Load XML and parse it
	tree = ET.parse(xml_file_path)
	root = tree.getroot()

	aam_text = root.find('.//MAPPING/AAM').text
	centre_text = root.find('.//FINGERPRINTS/CENTRE').text
	centre_text = centre_text[1:-1]
	#formed_cleaved_text = root.find('.//FINGERPRINTS/FORMED_CLEAVED').text
	fingerprints = root.find(".//FINGERPRINTS[@BC='1']")

	# tags under FINGERPRINTS[@BC='1'] is not fixed
	formed_cleaved_text = ""

	if fingerprints is not None:
		#we expect to find at least one child element under this <FINGERPRINTS> tag
		for child in fingerprints:
			formed_cleaved_text = child.text
			#print(f"Tag: {child.tag}, Text: {child.text}")
	else:
		print("No <FINGERPRINTS BC='1'> tag found.")
	
	# Find all chemical symbols in FORMED_CLEAVED
	formed_cleaved_symbols = set(re.findall(r'[A-Z][a-z]?', formed_cleaved_text))

	# Split the AAM text for substrate and product
	substrate_smiles, product_smiles = aam_text.split('>>')

	#parse SMILES strings
	parsed_smiles1 = parse_smiles(substrate_smiles)
	parsed_smiles2 = parse_smiles(product_smiles)

	#compare the parsed SMILES dictionaries
	soms = compare_smiles_dicts(parsed_smiles1, parsed_smiles2)
	
	return soms, substrate_smiles, product_smiles

    
