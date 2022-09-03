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
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("water_dimer.pdb")
    rc = 6
    # generator stores all force field parameters
    params = H.getParameters()
    
    pots = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
    pot_disp = pots.dmff_potentials['ADMPDispForce']
    pot_pme = pots.dmff_potentials['ADMPPmeForce']

    # construct inputs
    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10
    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T    

   
    # E_disp, F_disp = value_and_grad(pot_disp)(positions, box, pairs, disp_generator.params)
    # E_pme, F_pme = value_and_grad(pot_pme)(positions, box, pairs, pme_generator.params)

    # print('# Electrostatic+Polarization Energy:')
    # print('#', E_pme, 'kJ/mol')
    # print('# Dispersion+Damping Energy:')
    # print('#', E_disp, 'kJ/mol')
   

    # # compare induced dipole with mpid
    # with open('mpid_dip.pickle', 'rb') as ifile:
    #     U_ind_mpid = pickle.load(ifile) * 10

    # for x, y in zip(pme_generator.pme_force.U_ind.flatten(), U_ind_mpid.flatten()):
    #     print(y, y, x)

    param_grad = grad(pot_pme, argnums=(3))(positions, box, pairs, params)
    print(param_grad['ADMPPmeForce']['pScales'])
    print(param_grad['ADMPPmeForce']['dScales'])
