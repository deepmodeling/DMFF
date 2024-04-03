#!/usr/bin/env python
import time
import sys
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
import numpy as np
import dmff
from dmff.utils import jit_condition
# from gnn import MolGNNForce
from dmff.sgnn.gnn import MolGNNForce
# from graph import TopGraph, from_pdb
from dmff.sgnn.graph import TopGraph, from_pdb
import optax
import pickle
# use pytorch data loader
from torch.utils.data import DataLoader
from jax.lib import xla_bridge 
print(jax.devices()[0]) 
print(xla_bridge.get_backend().platform)


if __name__ == "__main__":
    box = jnp.eye(3) * 50
    pdbfile = 'PF6.pdb'
    # pdbfile = 'pdb_bank/BF4.pdb'
    # pdbfile = 'pdb_bank/DFP.pdb'
    # pdbfile = 'pdb_bank/FSI.pdb' # no problem
    G = from_pdb(pdbfile)
    model = MolGNNForce(G, nn=0, atype_index={'P': 0, 'F': 1}, max_valence=6)
    energy = model.forward(G.positions, box, model.params)
    print(energy)
