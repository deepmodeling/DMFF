#!/usr/bin/env python
import sys
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.common import nblist
from jax_md import space, partition
from jax import value_and_grad
import pickle

if __name__ == '__main__':
    
    H = Hamiltonian('forcefield.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("waterbox_31ang.pdb")
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
    nbl = nblist.NeighborList(box, rc, H.getGenerators()[0].covalent_map)
    nbl.allocate(positions)

   
    E_disp, F_disp = value_and_grad(pot_disp)(positions, box, nbl.pairs, params)
    E_pme, F_pme = value_and_grad(pot_pme)(positions, box, nbl.pairs, params)

    print('# Electrostatic+Polarization Energy:')
    print('#', E_pme, 'kJ/mol')
    print('# Dispersion+Damping Energy:')
    print('#', E_disp, 'kJ/mol')
   

