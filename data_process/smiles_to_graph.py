from rdkit import Chem
import networkx as nx
import math
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import os
import pandas as pd
import time
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from rdkit import RDLogger
from build_bg import construct_RGCN_bigraph_from_smiles


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold
# # test generate scaffold
# s = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
# scaffold = generate_scaffold(s)
# assert scaffold == 'c1ccc(Oc2ccccn2)cc1'

def scaffold_split(smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds. Deterministic split
    :param smiles_list: list of smiles
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :return: list of train, valid, test indices corresponding to the
    scaffold split
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    return train_idx, valid_idx, test_idx


class Dictionary(object):
    """
    worddidx is a dictionary
    idx2word is a list
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2num_occurence = {}
        self.idx2num_occurence = []

    def add_word(self, word):
        if word not in self.word2idx:
            # dictionaries
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            # stats
            self.idx2num_occurence.append(0)
            self.word2num_occurence[word] = 0

        # increase counters    
        self.word2num_occurence[word]+=1
        self.idx2num_occurence[  self.word2idx[word]  ] += 1

    def __len__(self):
        return len(self.idx2word)


def augment_dictionary(atom_dict, bond_dict, list_of_mol ):

    """
    take a lists of rdkit molecules and use it to augment existing atom and bond dictionaries
    """
    for idx,mol in enumerate(list_of_mol):

        for atom in mol.GetAtoms():
            atom_dict.add_word( atom.GetSymbol() )

        for bond in mol.GetBonds():
            bond_dict.add_word( str(bond.GetBondType()) )

        # compute the number of edges of type 'None'
        N=mol.GetNumAtoms()
        if N>2:
            E=N+math.factorial(N)/(math.factorial(2)*math.factorial(N-2)) # including self loop
            num_NONE_bonds = E-mol.GetNumBonds()
            bond_dict.word2num_occurence['NONE']+=num_NONE_bonds
            bond_dict.idx2num_occurence[0]+=num_NONE_bonds
def make_dictionary(list_of_mol):

    """
    the list of smiles (train, val and test) and build atoms and bond dictionaries
    """
    atom_dict=Dictionary()
    bond_dict=Dictionary()
    bond_dict.add_word('NONE')
    print('making dictionary')
    augment_dictionary(atom_dict, bond_dict, list_of_mol )
    print('complete')
    return atom_dict, bond_dict
import os

reg_root_file='you data path/reg/'
reg_files=os.listdir(reg_root_file)
clf_root_file='you data path/clf/'
clf_files=os.listdir(clf_root_file)
smiles_list=[]
for i in reg_files:
    path=os.path.join(reg_root_file+i)
    data=list(pd.read_csv(path)['smiles'])
    smiles_list.extend(data)
for i in clf_files:
    path=os.path.join(clf_root_file+i)
    data=list(pd.read_csv(path)['smiles'])
    smiles_list.extend(data)


chembl_data=pd.read_csv('your data path/chembl.csv')

smiles_list.extend(list(chembl_data['smiles'])) # get smiles strings from file

mol_list=[]
for smiles in smiles_list:
    try:
        mol=Chem.MolFromSmiles(smiles)
        if mol!=None:
            mol_list.append(mol)
    except:
        continue


atom_dict, bond_dict = make_dictionary(mol_list)




print(atom_dict.word2idx)
print(atom_dict.word2num_occurence)
print(bond_dict.word2idx)
print(bond_dict.word2num_occurence)

def mol_to_graph(mol,y,task,task_num):
    """
        mol is a rdkit mol object
    """
    no_bond_flag = False
    node_feat = torch.tensor([atom_dict.word2idx[atom.GetSymbol()] for atom in mol.GetAtoms()], dtype = torch.int64)
    N_atom=len(node_feat)
    #print(N_atom)
    if len(mol.GetBonds()) > 0: # mol has bonds
        bond_type=torch.zeros([N_atom,N_atom])
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_dict.word2idx[str(bond.GetBondType())]

            bond_type[i,j]=edge_feature
            bond_type[j,i]=edge_feature
        try:
            bg=construct_RGCN_bigraph_from_smiles(mol,atom_dict,bond_dict)
        except:
            no_bond_flag=True
        graph_dict=dict(
        num_atom=len(node_feat),
        atom_type=node_feat,
        bond_type=bond_type,
        y=y,
        bg=bg,
        task=task,
        task_num=task_num,
    )
        return graph_dict, no_bond_flag
    else:
        no_bond_flag = True
        return 0,no_bond_flag






## data processed for pretrain task

chembl_smiles=list(chembl_data['smiles']


                   

## data processed for admet task

reg_task=['LogS','LogD','LogP','MDCK','PPB','VDss','Fu','CL','Caco-2','IGC50','LC50DM','LC50','BCF']

clf_task=['T12','SR-p53','SR-MMP','SR-HSE','SR-ATAD5','SR-ARE','SkinSen','Respiratory','ROA','Pgp-sub','Pgp-inh',
'NR-PPAR-gamma','NR-ER','NR-AR-LBD','NR-Aromatase','NR-AR','NR-ER-LBD','NR-AhR','HIA','hERG','H-HT','FDAMDD',
'F(20%)','F(30%)','EI','EC','DILI','CYP3A4-sub','CYP3A4-inh','CYP2D6-sub','CYP2D6-inh','CYP2C19-inh',
'CYP2C19-sub','CYP2C9-sub','CYP2C9-inh','CYP1A2-inh','CYP1A2-sub','Carcinogenicity','BBB','Ames']

from sklearn.model_selection import train_test_split
train=[]
test=[]
val=[]

#####load  reg data set#####

for i in reg_files:
    path=os.path.join(reg_root_file+i)
    data=pd.read_csv(path)
    smiles_list=data['smiles']
    columns=data.columns
    label=data[columns[2]]
    task_num=reg_task.index(columns[2])
    graph=[]
    for i in range(len(smiles_list)):
        try:
            mol=Chem.MolFromSmiles(smiles_list[i])
            if mol!=None:
                graph_object, no_bond_flag = mol_to_graph(mol,label[i],0,task_num)       
        except:
            continue
        if no_bond_flag:
            continue
        else:
            graph.append(graph_object)

        train_mol, val_mol= train_test_split(graph,test_size=0.2, random_state=42)
        val_mol,test_mol=train_test_split(val_mol,test_size=0.5, random_state=42)

        train_path='./dataset/admet/'+columns[2]+'train.pickle'
        test_path='./dataset/admet/'+columns[2]+'test.pickle'
        val_path='./dataset/admet/'+columns[2]+'val.pickle'
        with open(train_path,'wb') as f:
            pickle.dump(train_mol,f)
        with open(val_path,'wb') as f:
            pickle.dump(val_mol,f)
        with open(test_path,'wb') as f:
            pickle.dump(test_mol,f)
        train.extend(train_mol)
        test.extend(test_mol)
        val.extend(val_mol)



#####load  clf data set#####
for i in clf_files:
    path=os.path.join(clf_root_file+i)
    data=pd.read_csv(path)
    smiles_list=data['smiles']
    columns=data.columns
    label=data[columns[2]]
    task_num=reg_task.index(columns[2])
    graph=[]
    for i in range(len(smiles_list)):
        try:
            mol=Chem.MolFromSmiles(smiles_list[i])
            if mol!=None:
                graph_object, no_bond_flag = mol_to_graph(mol,label[i],1,task_num)       
        except:
            continue
        if no_bond_flag:
            continue
        else:
            graph.append(graph_object)

        train_mol, val_mol= train_test_split(graph,test_size=0.2, random_state=42)
        val_mol,test_mol=train_test_split(val_mol,test_size=0.5, random_state=42)

        train_path='./dataset/admet/'+columns[2]+'train.pickle'
        test_path='./dataset/admet/'+columns[2]+'test.pickle'
        val_path='./dataset/admet/'+columns[2]+'val.pickle'
        with open(train_path,'wb') as f:
            pickle.dump(train_mol,f)
        with open(val_path,'wb') as f:
            pickle.dump(val_mol,f)
        with open(test_path,'wb') as f:
            pickle.dump(test_mol,f)
        train.extend(train_mol)
        test.extend(test_mol)
        val.extend(val_mol)




