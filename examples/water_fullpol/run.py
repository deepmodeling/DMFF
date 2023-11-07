#!/usr/bin/env python
import sys
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.common import nblist
from jax import value_and_grad
import pickle

if __name__ == '__main__':
    
    H = Hamiltonian('forcefield.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("waterbox_31ang.pdb")
    rc = 0.6
    # generator stores all force field parameters
    disp_generator, pme_generator = H.getGenerators()
    
    pots = H.createPotential(pdb.topology, \
                             nonbondedMethod=app.PME, \
                             nonbondedCutoff=rc*unit.nanometer, \
                             has_aux=True, \
                             ethresh=5e-4)

    # construct inputs
    positions = jnp.array(pdb.positions._value)
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value])
    # neighbor list
    nbl = nblist.NeighborList(box, rc, pots.meta['cov_map']) 
    nbl.allocate(positions)

  
    params = H.getParameters()

    pot_disp = pots.dmff_potentials['ADMPDispForce']
    res_disp, F_disp = value_and_grad(pot_disp, has_aux=True)(positions, box, nbl.pairs, params)
    E_disp = res_disp[0]
    pot_pme = pots.dmff_potentials['ADMPPmeForce']
    res_pme, F_pme = value_and_grad(pot_pme, has_aux=True)(positions, box, nbl.pairs, params)
    E_pme = res_pme[0]
    U_ind = res_pme[1]["U_ind"]

    print('# Electrostatic+Polarization Energy:')
    print('#', E_pme, 'kJ/mol')
    print('# Dispersion+Damping Energy:')
    print('#', E_disp, 'kJ/mol')

    # compare induced dipole with mpid
    with open('mpid_dip.pickle', 'rb') as ifile:
        U_ind_mpid = pickle.load(ifile) * 10

    for x, y in zip(U_ind.flatten(), U_ind_mpid.flatten()):
        print(y, y, x)
