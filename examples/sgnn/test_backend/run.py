#!/usr/bin/env python
import sys
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap, value_and_grad
import dmff
from dmff.sgnn.gnn import MolGNNForce
from dmff.utils import jit_condition
from dmff.sgnn.graph import MAX_VALENCE
from dmff.sgnn.graph import TopGraph, from_pdb
import pickle
import re
from collections import OrderedDict
from functools import partial


if __name__ == '__main__':
    # params = load_params('benchmark/model1.pickle')
    G = from_pdb('peg4.pdb')
    model = MolGNNForce(G, nn=1)
    model.load_params('model1.pickle')
    E = model.get_energy(G.positions, G.box, model.params)

    with open('set_test_lowT.pickle', 'rb') as ifile:
        data = pickle.load(ifile)

    # pos = jnp.array(data['positions'][0:100])
    # box = jnp.tile(jnp.eye(3) * 50, (100, 1, 1))
    pos = jnp.array(data['positions'][0])
    box = jnp.eye(3) * 50

    # energies = model.batch_forward(pos, box, model.params)
    E, F = value_and_grad(model.get_energy, argnums=(0))(pos, box, model.params)
    F = -F
    print('Energy:', E)
    print('Force')
    print(F)

    # test batch processing
    pos = jnp.array(data['positions'][:20])
    box = jnp.tile(box, (20, 1, 1))
    E = model.batch_forward(pos, box, model.params)
    print('Batched Energies:')
    print(E)
