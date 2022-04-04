#!/usr/bin/env python
import sys
from pathlib import Path
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
admp_path = Path(__file__).parent.parent.parent
sys.path.append(str(admp_path))
import numpy as np
import jax.numpy as jnp
from jax_md import partition, space
from dmff.admp.multipole import convert_cart2harm
from dmff.admp.pme import ADMPPmeForce
from dmff.admp.parser import *
from jax import grad


import linecache
def get_line_context(file_path, line_number):
    return linecache.getline(file_path,line_number).strip()

# below is the validation code
if __name__ == '__main__':
    
    rc = 4.0
    
    H = Hamiltonian('forcefield.xml')
    app.Topology.loadBondDefinitions('residues.xml')
    pdb = app.PDBFile('waterbox_31ang.pdb')
    
    generator = H.getGenerators()
    disp_generator, pme_generator = generator
    
    pme_generator.lpol = True
    pme_generator.ref_dip = 'dipole_1024'
    potentials = H.createPotential(pdb.topology, nonbondedCutoff=4.0*unit.angstrom)
    
    disp_pot, pme_pot = potentials
    
    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10
    
    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T

    n_atoms = len(positions)

    # construct the C list
    c_list = np.zeros((3, n_atoms))
    a_list = np.zeros(n_atoms)
    q_list = np.zeros(n_atoms)
    b_list = np.zeros(n_atoms)
    nmol=int(n_atoms/3)
    for i in range(nmol):
        a = i*3
        b = i*3+1
        c = i*3+2
        # dispersion coeff
        c_list[0][a]=37.19677405
        c_list[0][b]=7.6111103
        c_list[0][c]=7.6111103
        c_list[1][a]=85.26810658
        c_list[1][b]=11.90220148
        c_list[1][c]=11.90220148
        c_list[2][a]=134.44874488
        c_list[2][b]=15.05074749
        c_list[2][c]=15.05074749
        # q
        q_list[a] = -0.741706
        q_list[b] = 0.370853
        q_list[c] = 0.370853
        # b, Bohr^-1
        b_list[a] = 2.00095977
        b_list[b] = 1.999519942
        b_list[c] = 1.999519942
        # a, Hartree
        a_list[a] = 458.3777
        a_list[b] = 0.0317
        a_list[c] = 0.0317

    # Finish data preparation
    # -------------------------------------------------------------------------------------
    # parameters should be ready: 
    # geometric variables: positions, box
    # atomic parameters: Q_local, c_list
    # topological parameters: covalent_map, mScales, pScales, dScales
    # general force field setting parameters: rc, ethresh, lmax, pmax


    # get neighbor list using jax_md
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T

    # electrostatic
    pme_force = ADMPPmeForce(box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax, lpol=True)
    pme_force.update_env('kappa', 0.657065221219616)
    pot_pme = pme_force.get_energy
    jnp.save('mScales', mScales)
    jnp.save('Q_local', Q_local)
    jnp.save('pol', pol)
    jnp.save('tholes', tholes)
    jnp.save('pScales', pScales)
    jnp.save('dScales', dScales)
    jnp.save('U_ind', pme_force.U_ind)  
    # E, F = pme_force.get_forces(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales)
    # print('# Electrostatic Energy (kJ/mol)')
    # E = pme_force.get_energy(positions, box, pairs, Q_local, mScales, pScales, dScales)
    E = pot_pme(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, U_init=pme_force.U_ind)
    grad_params = grad(pot_pme, argnums=(3,4,5,6,7,8,9))(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, pme_force.U_ind)
    # print(E)
    U_ind = pme_force.U_ind
    # compare U_ind with reference
    for i in range(1024):
        for j in range(3):
            print(Uind_global[i*3, j], Uind_global[i*3, j], U_ind[i*3, j])

