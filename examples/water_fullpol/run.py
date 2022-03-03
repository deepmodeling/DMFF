#!/usr/bin/env python
import sys
import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad
from jax_md import partition, space
import dmff
from dmff.admp.settings import *
from dmff.admp.multipole import convert_cart2harm
from dmff.admp.pme import ADMPPmeForce
from dmff.admp.parser import *
from dmff.admp.disp_pme import ADMPDispPmeForce
from dmff.admp.pairwise import generate_pairwise_interaction, TT_damping_qq_c6_kernel
import pickle


import linecache
def get_line_context(file_path, line_number):
    return linecache.getline(file_path,line_number).strip()


# below is the validation code
if __name__ == '__main__':
    pdb = 'waterbox_31ang.pdb'
    xml = 'fullpol.xml'
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
    dScales = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])

    rc = 4  # in Angstrom
    ethresh = 1e-4

    n_atoms = len(serials)

    # compute intra
    #E1 = onebodyenergy(n_atoms, positions)
    #print(E1)

    atomTemplate, residueTemplate = read_xml(xml)
    atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

    Q = np.vstack(
        [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
    )

    c0 = []
    c6_list = []
    #compute geometry-dependent terms
    b=[np.arange(n_atoms)[i:i+3] for i in range(0,len(np.arange(n_atoms)),3)]
    for i in b:
        O = positions[i][0]
        H1 = positions[i][1]
        H2 = positions[i][2]
        bond1_len = (np.linalg.norm(H1-O))
        bond2_len = (np.linalg.norm(H2-O))
        bond1 = H1-O
        bond2 = H2-O
        cos_angle = np.dot(bond1,bond2)/(bond1_len * bond2_len)
        angle = np.arccos(cos_angle)*180/np.pi 
        dipole = -0.016858755+0.002287251*angle + 0.239667591*bond1_len + (-0.070483437)*bond2_len
        charge_H = dipole/bond1_len 
        charge_O=charge_H*(-2)
        C6_H = (-2.36066199 + (-0.007049238)*angle + 1.949429648*bond1_len + 2.097120784*bond2_len) * 0.529**6 * 2625.5
        C6_O = (-8.641301261 + 0.093247893*angle + 11.90395358*(bond1_len+bond2_len)) * 0.529**6 * 2625.5
        c0.append(charge_O)
        c0.append(charge_H)
        c0.append(charge_H)
        c6_list.append(np.sqrt(C6_O))
        c6_list.append(np.sqrt(C6_H))
        c6_list.append(np.sqrt(C6_H))


    # change leading term
    #Q[:,0]=c0

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
        # b, Bohr^-1
        b_list[a] = 2.00095977
        b_list[b] = 1.999519942
        b_list[c] = 1.999519942
        # a, Hartree
        a_list[a] = 0
        a_list[b] = 0
        a_list[c] = 0
    
    # change the leading term
    c_list[0]=c6_list
    c_list = jnp.array(c_list.T)
    q_list = c0
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


    pme_force = ADMPPmeForce(box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax, lpol=True)
    # import pickle
    # with open('mpid_dip.pickle', 'rb') as ifile:
    #     U_ind = jnp.array(pickle.load(ifile) * 10)
    # U_ind = jnp.zeros(U_ind.shape)
    U_ind = None
    E, F = pme_force.get_forces(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, U_init=U_ind)
    print('# Polarizable Electrostatic Energy (kJ/mol)')
    print('#', E)

    # compare induced dipole with mpid
    with open('mpid_dip.pickle', 'rb') as ifile:
        U_ind_mpid = pickle.load(ifile) * 10

    for x, y in zip(pme_force.U_ind.flatten(), U_ind_mpid.flatten()):
        print(y, y, x)

    
