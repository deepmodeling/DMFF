#!/usr/bin/env python
import sys
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from jax_md import space, partition
from jax import value_and_grad, grad
import pickle

if __name__ == '__main__':
    
    H = Hamiltonian('forcefield.xml')
    pdb = app.PDBFile("waterbox_31ang.pdb")
    rc = 6
    # generator stores all force field parameters
    pme_generator = H.getGenerators()[1]
    
    pot_pme = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)[1]

    # construct inputs
    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10
    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T    

    E_pme, F_pme = value_and_grad(pot_pme)(positions, box, pairs, pme_generator.params)

    print('Electrostatic Energy (kJ/mol):')
    print(E_pme)

