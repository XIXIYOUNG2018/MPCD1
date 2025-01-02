from dgl import DGLGraph
import pandas as pd
from rdkit.Chem import MolFromSmiles
import numpy as np
from rdkit import Chem
from dgl.data.graph_serialize import save_graphs
import torch
import rdkit



def construct_RGCN_bigraph_from_smiles(mol,atom_dict,bond_dict):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    atoms_feature_all = []
    for atom in mol.GetAtoms():
        atom_feature = atom_dict.word2idx[atom.GetSymbol()]
        atoms_feature_all.append(atom_feature)
    g.ndata["atom"] = torch.tensor(atoms_feature_all)

    # Add edges
    src_list = []
    dst_list = []
    etype_feature_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = bond_dict.word2idx[str(bond.GetBondType())]
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.append(etype_feature)
        etype_feature_all.append(etype_feature)

    g.add_edges(src_list, dst_list)
    normal_all = []
    for i in etype_feature_all:
        normal = etype_feature_all.count(i)/len(etype_feature_all)
        normal = round(normal, 1)
        normal_all.append(normal)

    g.edata["etype"] = torch.tensor(etype_feature_all)
    g.edata["normal"] = torch.tensor(normal_all)
    return g



   























