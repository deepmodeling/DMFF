#!/usr/bin/env python
import sys
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from jax_md import space, partition
from jax import value_and_grad, jit
import pickle

def admp_calculator(positions, box, pairs, disp_params, pme_params):
    E_disp = pot_disp(positions, box, pairs, disp_params)
    E_pme = pot_pme(positions, box, pairs, pme_params)
    return E_disp + E_pme

if __name__ == '__main__':
    
    H = Hamiltonian('forcefield.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("water_dimer.pdb")
    rc = 4
    # generator stores all force field parameters
    disp_generator, pme_generator = H.getGenerators()
    
    pot_disp, pot_pme = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom, step_pol=5)

    # construct inputs
    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10
    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T    

    admp_calc = jit(value_and_grad(admp_calculator,argnums=(0)))
    tot_ene, tot_force = admp_calc(positions, box, pairs, disp_generator.params, pme_generator.params)
    print(tot_ene)
    print(tot_force)
