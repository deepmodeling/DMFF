#!/usr/bin/env python
import sys
from pathlib import Path
admp_path = Path(__file__).parent.parent.parent
sys.path.append(str(admp_path))
import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad
from jax_md import partition, space
from dmff.admp.multipole import convert_cart2harm
from dmff.admp.pme import ADMPPmeForce
from dmff.admp.disp_pme import ADMPDispPmeForce
from dmff.admp.pairwise import generate_pairwise_interaction, TT_damping_qq_c6_kernel
from dmff.admp.parser import *


# below is the validation code
if __name__ == '__main__':
    pdb = 'water1024.pdb'
    xml = 'mpidwater.xml'
    pdbinfo = read_pdb(pdb)
    serials = pdbinfo['serials']
    names = pdbinfo['names']
    resNames = pdbinfo['resNames']
    resSeqs = pdbinfo['resSeqs']
    positions = pdbinfo['positions']
    box = pdbinfo['box'] # a, b, c, α, β, γ
    charges = pdbinfo['charges']
    positions = jnp.asarray(positions)
    lx, ly, lz, _, _, _ = box
    box = jnp.eye(3)*jnp.array([lx, ly, lz])

    mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

    rc = 4  # in Angstrom
    ethresh = 1e-4

    n_atoms = len(serials)

    atomTemplate, residueTemplate = read_xml(xml)
    atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

    Q = np.vstack(
        [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
    )
    Q = jnp.array(Q)
    Q_local = convert_cart2harm(Q, 2)
    axis_type = np.array(
        [atom.axisType for atom in atomDicts.values()]
    )
    axis_indices = np.vstack(
        [atom.axis_indices for atom in atomDicts.values()]
    )
    covalent_map = assemble_covalent(residueDicts, n_atoms)

    
    lmax = 2
    pmax = 10

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
    pme_force = ADMPPmeForce(box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax)
    pme_force.update_env('kappa', 0.657065221219616)
    E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales)
    print('Electrostatic Energy (kJ/mol)')
    # E = pme_force.get_energy(positions, box, pairs, Q_local, mScales, pScales, dScales)
    E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales)
    print(E)


    # dispersion
    disp_pme_force = ADMPDispPmeForce(box, covalent_map, rc, ethresh, pmax)
    disp_pme_force.update_env('kappa', 0.657065221219616)
    E, F = disp_pme_force.get_forces(positions, box, pairs, c_list.T, mScales)
    print('Dispersion Energy (kJ/mol)')
    # E = disp_pme_force.get_energy(positions, box, pairs, c_list.T, mScales)
    E, F = disp_pme_force.get_forces(positions, box, pairs, c_list.T, mScales)
    print(E)
   

    # short range damping
    TT_damping_qq_c6 = value_and_grad(generate_pairwise_interaction(TT_damping_qq_c6_kernel, covalent_map, static_args={}))
    TT_damping_qq_c6(positions, box, pairs, mScales, a_list, b_list, q_list, c_list[0])
    print('Tang-Tonnies Damping (kJ/mol)')
    E, F = TT_damping_qq_c6(positions, box, pairs, mScales, a_list, b_list, q_list, c_list[0])
    print(E)

