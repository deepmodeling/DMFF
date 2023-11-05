#!/usr/bin/env python
import sys
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from jax_md import space, partition
from jax import value_and_grad, grad
from dmff.common import nblist
import pickle

if __name__ == '__main__':
    
    H = Hamiltonian('forcefield.xml')
    pdb = app.PDBFile("pair.pdb")
    rc = 0.6
    # generator stores all force field parameters
    params = H.getParameters()
    
    pots = H.createPotential(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer)
    pot_pme = pots.dmff_potentials['ADMPPmeForce']

    # construct inputs
    positions = jnp.array(pdb.positions._value)
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value])
    # neighbor list
    nbl = nblist.NeighborList(box, rc, pots.meta['cov_map'])
    nbl.allocate(positions)

    E_pme, F_pme = value_and_grad(pot_pme)(positions, box, nbl.pairs, params)

    print('Electrostatic Energy (kJ/mol):')
    print(E_pme)

