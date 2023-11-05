#!/usr/bin/env python
import sys
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.common import nblist
from jax import value_and_grad
import pickle

if __name__ == '__main__':
    
    H = Hamiltonian('peg.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("peg4.pdb")
    rc = 0.4
    # generator stores all force field parameters
    pots = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.nanometer, ethresh=5e-4)

    # construct inputs
    positions = jnp.array(pdb.positions._value)
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value])
    # neighbor list
    nbl = nblist.NeighborList(box, rc, pots.meta['cov_map']) 
    nbl.allocate(positions)

  
    paramset = H.getParameters()
    # params = paramset.parameters
    paramset.parameters

    efunc = jax.jit(pots.getPotentialFunc())
    print(efunc(positions, box, nbl.pairs, paramset))

