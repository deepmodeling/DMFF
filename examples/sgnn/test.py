#!/usr/bin/env python
import sys
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
import numpy as np
import dmff
from dmff.utils import jit_condition
from dmff.sgnn.gnn import MolGNNForce
from dmff.sgnn.graph import TopGraph, from_pdb
import optax
import pickle
# use pytorch data loader
from torch.utils.data import DataLoader

class MolDataSet():

    def __init__(self, pdb, pickle_fn):
        self.file = pickle_fn
        with open(pickle_fn, 'rb') as f:
            self.data = pickle.load(f)
        self.n_data = len(self.data['positions'])
        self.pickle = pickle_fn
        self.pdb = pdb
        return

    def __getitem__(self, i):
        return [self.data['positions'][i], self.data['energies'][i]]

    def __len__(self):
        return self.n_data


if __name__ == "__main__":

    # training and testing data
    dataset = MolDataSet('peg4.pdb', 'set009_remove_nb2.pickle')
    loader = DataLoader(dataset, shuffle=True, batch_size=100)
    box = jnp.eye(3) * 50

    # Graph and model
    G = from_pdb('peg4.pdb')
    model = MolGNNForce(G, nn=1)
    model.batch_forward = jax.vmap(model.forward, in_axes=(0, None, None), out_axes=(0))
    model.load_params(sys.argv[1])

    # evaluate test
    ene_refs = []
    ene_preds = []
    for pos, e in loader:
        ene_ref = jnp.array(e.numpy())
        pos = jnp.array(pos.numpy())
        ene_pred = model.batch_forward(pos, box, model.params)
        ene_preds.append(ene_pred)
        ene_refs.append(ene_ref)
    ene_ref = jnp.concatenate(ene_refs)
    ene_ref = ene_ref - jnp.average(ene_ref)
    ene_pred = jnp.concatenate(ene_preds)
    ene_pred = ene_pred - jnp.average(ene_pred)
    err = ene_pred - ene_ref
    loss = jnp.average(err**2)
    print(loss)
