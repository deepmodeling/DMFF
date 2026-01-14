#!/usr/bin/env python
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.common import nblist
import numpy as np
import jax
import jax.numpy as jnp
import torch
from torch2jax import j2t, t2j
# from dmff.torch_tools import wrap_torch_potential_kernel, j2t_pytree, t2j_pytree

from force_model import ForceModel


if __name__ == '__main__':
    H = Hamiltonian('forcefield.xml')
    params = H.getParameters().parameters
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("structure.pdb")
    rc = 1.0

    pots = H.createPotential(pdb.topology, \
                             nonbondedMethod=app.CutoffPeriodic, \
                             nonbondedCutoff=rc*unit.nanometer)

    efunc = pots.getPotentialFunc()
    positions = jnp.array(pdb.positions._value)
    box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value)

    ene, pgrad = jax.value_and_grad(efunc, argnums=3)(positions, box, None, params)
    ene, rgrad = jax.value_and_grad(efunc, argnums=0)(positions, box, None, params)
    print('Predicted Energy:')
    print(ene)
    print('Parameter Gradient:')
    print(pgrad)
    print('Position Gradient:')
    print(rgrad)

    # saving new parameters
    gen = H.getGenerators()[0]
    gen.write_to(params, 'new.pt', 'new_sd.pt')
