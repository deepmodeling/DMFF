#!/usr/bin/env python
import sys
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
import numpy as np
import dmff
from dmff.utils import jit_condition
from dmff.sgnn.gnn import MolGNN
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
    restart = None # 'model1.pickle'

    # training and testing data
    dataset = MolDataSet('peg4.pdb', 'set_train.pickle')
    train_loader = DataLoader(dataset, shuffle=True, batch_size=8)
    dataset_test = MolDataSet('peg4.pdb', 'set_test.pickle')
    test_loader = DataLoader(dataset_test, batch_size=100)
    box = jnp.eye(3) * 50

    # Graph and model
    G = from_pdb('peg4.pdb')
    model = MolGNN(G, nn=1)
    model.batch_forward = jax.vmap(model.forward, in_axes=(0, None, None), out_axes=(0))
    if restart is not None:
        model.load_params(restart)

    # optmizer
    optimizer = optax.adam(0.0001)
    opt_state = optimizer.init(model.params)

    # mean square loss function
    def MSELoss(params, positions, box, ene_ref):
        ene = model.batch_forward(positions, box, params)
        err = ene - ene_ref
        # we do not care about constant shifts
        err -= jnp.average(err)
        return jnp.average(err**2)
    MSELoss = jit(MSELoss)

    # train
    n_epochs = 2000
    iprint = 0
    for i_epoch in range(n_epochs):
        # train an epoch
        for ibatch, (pos, e) in enumerate(train_loader):
            pos = jnp.array(pos.numpy())
            ene_ref = jnp.array(e.numpy())
            loss, gradients = value_and_grad(MSELoss, argnums=(0))(model.params, pos, box, ene_ref)
            updates, opt_state = optimizer.update(gradients, opt_state)
            model.params = optax.apply_updates(model.params, updates)
            print(loss)
            iprint += 1
            sys.stdout.flush()

        # save model after each epoch
        model.save_params('model.pickle')

        # evaluate test
        ene_refs = []
        ene_preds = []
        for pos, e in test_loader:
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
        # print test loss
        with open('mse_testing.xvg', 'a') as f:
            print(iprint, loss, file=f)
        # print test data
        with open('test_data.xvg', 'w') as f:
            for e1, e2 in zip(ene_pred, ene_ref):
                print(e2, e2, e1, file=f)
            
