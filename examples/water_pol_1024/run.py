#!/usr/bin/env python
import sys
from pathlib import Path
admp_path = Path(__file__).parent.parent.parent
sys.path.append(str(admp_path))
import numpy as np
import jax.numpy as jnp
from jax_md import partition, space
from dmff.admp.multipole import convert_cart2harm
from dmff.admp.pme import ADMPPmeForce
from dmff.admp.parser import *
from dmff.admp.disp_pme import ADMPDispPmeForce
from dmff.admp.pairwise import generate_pairwise_interaction, TT_damping_qq_c6_kernel
from dmff.admp.intra import *
from jax import grad, value_and_grad
import time
from dmff.admp.spatial import v_pbc_shift
import linecache
def get_line_context(file_path, line_number):
    return linecache.getline(file_path,line_number).strip()

# below is the validation code
if __name__ == '__main__':
    pdb = str(sys.argv[1])
    xml = str(sys.argv[2])
    #ref_dip = str('dipole_1024')
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

    c0 = np.zeros(n_atoms)
    c6_list = np.zeros(n_atoms)
    #compute geometry-dependent terms
    box_inv = jnp.linalg.inv(box)
    O = positions[::3]
    H1 = positions[1::3]
    H2 = positions[2::3]
    ROH1 = H1 - O
    ROH2 = H2 - O
    ROH1 = v_pbc_shift(ROH1, box, box_inv)
    ROH2 = v_pbc_shift(ROH2, box, box_inv)
    dROH1 = np.linalg.norm(ROH1, axis=1)
    dROH2 = np.linalg.norm(ROH2, axis=1)
    costh = np.sum(ROH1 * ROH2, axis=1) / (dROH1 * dROH2)
    angle = np.arccos(costh)*180/np.pi
    dipole = -0.016858755+0.002287251*angle + 0.239667591*dROH1 + (-0.070483437)*dROH2
    charge_H = dipole/dROH1
    charge_O=charge_H*(-2)
    C6_H = (-2.36066199 + (-0.007049238)*angle + 1.949429648*dROH1+ 2.097120784*dROH2) * 0.529**6 * 2625.5
    C6_O = (-8.641301261 + 0.093247893*angle + 11.90395358*(dROH1+ dROH2)) * 0.529**6 * 2625.5
    c0[::3] = charge_O
    c0[1::3] = charge_H
    c0[2::3] = charge_H
    c6_list[::3] = np.sqrt(C6_O)
    c6_list[1::3] = np.sqrt(C6_H)
    c6_list[2::3] = np.sqrt(C6_H)
    
    

    # change leading term
    Q[:,0]=c0

    Q = jnp.array(Q)
    Q_local = convert_cart2harm(Q, 2)
    axis_type = np.array(
        [atom.axisType for atom in atomDicts.values()]
    )
    axis_indices = np.vstack(
        [atom.axis_indices for atom in atomDicts.values()]
    )
    covalent_map = assemble_covalent(residueDicts, n_atoms)

    ## ind paras
    pol = np.vstack(
        [(atom.polarizabilityXX, atom.polarizabilityYY, atom.polarizabilityZZ) for atom in atomDicts.values()]
    )
    pol = jnp.array(pol.astype(np.float32))
    pol = 1000*jnp.mean(pol,axis=1)

    tholes = np.vstack(
        [atom.thole  for atom in atomDicts.values()]
    )
    tholes = jnp.array(tholes.astype(np.float32))
    tholes = jnp.mean(tholes,axis=1) 
    defaultTholeWidth=8
   


    
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
        a_list[a] = 72.02844
        a_list[b] = 2.3870113
        a_list[c] = 2.3870113

    c_list[0]=c6_list
    c_list = jnp.array(c_list.T)
    q_list = jnp.array(c0)
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
    E, F = pme_force.get_forces(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales)
    print('# Electrostatic Energy (kJ/mol)')
    #E = pme_force.get_energy(positions, box, pairs, Q_local, mScales, pScales, dScales)
    print(E)
    E = pot_pme(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, U_init=pme_force.U_ind)
    
    grad_params = grad(pot_pme, argnums=(3,4,5,6,7,8,9))(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, pme_force.U_ind)
    U_ind = pme_force.U_ind
    
    # dispersion
    disp_pme_force = ADMPDispPmeForce(box, covalent_map, rc, ethresh, pmax)
    disp_pme_force.update_env('kappa', 0.657065221219616)
    E, F = disp_pme_force.get_forces(positions, box, pairs, c_list, mScales)
    print('Dispersion Energy (kJ/mol)')
    #E = disp_pme_force.get_energy(positions, box, pairs, c_list.T, mScales)
    #E, F = disp_pme_force.get_forces(positions, box, pairs, c_list.T, mScales)
    print(E)

    # short range damping
    c6_list = jnp.array(c6_list)
    TT_damping_qq_c6 = value_and_grad(generate_pairwise_interaction(TT_damping_qq_c6_kernel, covalent_map, static_args={}))
    TT_damping_qq_c6(positions, box, pairs, mScales, a_list, b_list, q_list, c6_list)
    print('Tang-Tonnies Damping (kJ/mol)')
    E, F = TT_damping_qq_c6(positions, box, pairs, mScales, a_list, b_list, q_list, c6_list)
    print(E)

    # intramolecular term
    print('Intramolecular Energy (kJ/mol)')
    E1 = onebodyenergy(positions, box)
    #force = grad_E1(n_atoms,positions, box)
    print(E1)       
