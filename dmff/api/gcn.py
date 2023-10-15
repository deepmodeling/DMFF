from typing import List, Tuple
from pathlib import Path
import numpy as np
from .topology import DMFFTopology


EMBED_W1 = np.random.random((117, 117))

elem_to_index = {'EP': 0, 'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 
                 'F': 9, 'NE': 10, 'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 
                 'CL': 17, 'AR': 18, 'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 
                 'MN': 25, 'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30, 'GA': 31, 'GE': 32, 
                 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36, 'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 
                 'NB': 41, 'MO': 42, 'TC': 43, 'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48, 
                 'IN': 49, 'SN': 50, 'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 
                 'LA': 57, 'CE': 58, 'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64, 
                 'TB': 65, 'DY': 66, 'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72, 
                 'TA': 73, 'W': 74, 'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80, 
                 'TL': 81, 'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86, 'FR': 87, 'RA': 88, 
                 'AC': 89, 'TH': 90, 'PA': 91, 'U': 92, 'NP': 93, 'PU': 94, 'AM': 95, 'CM': 96, 
                 'BK': 97, 'CF': 98, 'ES': 99, 'FM': 100, 'MD': 101, 'NO': 102, 'LR': 103, 
                 'RF': 104, 'DB': 105, 'SG': 106, 'BH': 107, 'HS': 108, 'MT': 109, 'DS': 110, 
                 'RG': 111, 'UUB': 112, 'UUT': 113, 'UUQ': 114, 'UUP': 115, 'UUH': 116}


def mol_to_graph_matrix(topdata: DMFFTopology) -> Tuple[np.ndarray, np.ndarray]:
    num_atom = topdata.getNumAtoms()
    adj = np.zeros((num_atom, num_atom))
    node_features = np.zeros((num_atom, 117))
    for i in range(num_atom):
        adj[i, i] = 1
        node_features[i, elem_to_index[topdata._atom[i].element.upper()]] = 1
    for bond in topdata.bonds():
        adj[bond.atom1.index, bond.atom2.index] = 1
        adj[bond.atom2.index, bond.atom1.index] = 1
    return adj, node_features


def get_embed(topdata: DMFFTopology):
    adj, node = mol_to_graph_matrix(topdata)
    natom = adj.shape[0]
    support = np.dot(node[:, :117], EMBED_W1[:117, :117])
    out = np.dot(adj, support)
    out = np.concatenate((out, node[:, 117:]), axis=1)
    return out


def get_eqv_atoms(topdata: DMFFTopology):
    embed = get_embed(topdata)
    natom, nfeat = embed.shape[0], embed.shape[1]
    dist = np.power(
        embed.reshape((natom, 1, nfeat)) - embed.reshape((1, natom, nfeat)), 2
    ).sum(axis=2)
    eqv_list = []
    for na in range(natom):
        eqv_list.append([na])
        for nb in range(natom):
            if dist[na, nb] < 1e-2 and na != nb:
                eqv_list[-1].append(nb)
    return eqv_list

