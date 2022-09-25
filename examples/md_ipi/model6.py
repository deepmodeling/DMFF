#!/usr/bin/env python
import sys
import numpy as np
import jax.numpy as jnp
from jax_md import partition, space
from dmff.utils import jit_condition
from dmff.admp.multipole import convert_cart2harm
from dmff.admp.pme import ADMPPmeForce
from dmff.admp.parser import *
from dmff.admp.disp_pme import ADMPDispPmeForce
from dmff.admp.pairwise import generate_pairwise_interaction, TT_damping_qq_c6_kernel
from intra import onebodyenergy
from jax import grad, value_and_grad
import time
from dmff.admp.spatial import v_pbc_shift
import linecache
from dmff.settings import DO_JIT
from jax import grad, value_and_grad, vmap, jit
def get_line_context(file_path, line_number):
    return linecache.getline(file_path,line_number).strip()

def gen_trim_val_0(thresh):
    '''
    Trim the value at zero point to avoid singularity
    '''
    def trim_val_0(x):
        return jnp.piecewise(x, [x<thresh, x>=thresh], [lambda x: jnp.array(thresh), lambda x: x])
    if DO_JIT:
        return jit(trim_val_0)
    else:
        return trim_val_0

trim_val_0 = gen_trim_val_0(1e-8)

@jit_condition(static_argnums=())
def compute_leading_terms(positions,box):
    n_atoms = len(positions)
    c0 = jnp.zeros(n_atoms)
    c6_list = jnp.zeros(n_atoms)
    box_inv = jnp.linalg.inv(box)
    O = positions[::3]
    H1 = positions[1::3]
    H2 = positions[2::3]
    ROH1 = H1 - O
    ROH2 = H2 - O
    ROH1 = v_pbc_shift(ROH1, box, box_inv)
    ROH2 = v_pbc_shift(ROH2, box, box_inv)
    dROH1 = jnp.linalg.norm(ROH1, axis=1)
    dROH2 = jnp.linalg.norm(ROH2, axis=1)
    costh = jnp.sum(ROH1 * ROH2, axis=1) / (dROH1 * dROH2)
    angle = jnp.arccos(costh)*180/jnp.pi
    dipole1 = -0.016858755+0.002287251*angle + 0.239667591*dROH1 + (-0.070483437)*dROH2
    charge_H1 = dipole1/dROH1
    dipole2 = -0.016858755+0.002287251*angle + 0.239667591*dROH2 + (-0.070483437)*dROH1
    charge_H2 = dipole2/dROH2
    charge_O = -(charge_H1 + charge_H2)
    C6_H1 = (-2.36066199 + (-0.007049238)*angle + 1.949429648*dROH1+ 2.097120784*dROH2) * 0.529**6 * 2625.5
    C6_H2 = (-2.36066199 + (-0.007049238)*angle + 1.949429648*dROH2+ 2.097120784*dROH1) * 0.529**6 * 2625.5
    C6_O = (-8.641301261 + 0.093247893*angle + 11.90395358*(dROH1+ dROH2)) * 0.529**6 * 2625.5
    C6_H1 = trim_val_0(C6_H1)
    C6_H2 = trim_val_0(C6_H2)
    c0 = c0.at[::3].set(charge_O)
    c0 = c0.at[1::3].set(charge_H1)
    c0 = c0.at[2::3].set(charge_H2)
    c6_list = c6_list.at[::3].set(jnp.sqrt(C6_O))
    c6_list = c6_list.at[1::3].set(jnp.sqrt(C6_H1))
    c6_list = c6_list.at[2::3].set(jnp.sqrt(C6_H2))
    return c0, c6_list

def read_input_info(pdb,xml):
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
        a_list[a] = 184.99965
        a_list[b] = 0.1324176
        a_list[c] = 0.1324176
    c_list = jnp.array(c_list.T) 
    return positions, Q_local, axis_type, axis_indices, box, covalent_map, pol, tholes, c_list, q_list, a_list, b_list

def admp_calculator(positions, Q_local, axis_type, axis_indices, box, covalent_map, pol, tholes, c_list, q_list, a_list, b_list, pairs):
    c0, c6_list = compute_leading_terms(positions, box)
    Q_local = Q_local.at[:,0].set(c0)
    c_list = c_list.at[:,0].set(c6_list)
    q_list = c0
    
    E1 = pme_force.get_energy(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales)
    E2 = disp_pme_force.get_energy(positions, box, pairs, c_list, mScales)
    E3 = TT_damping_qq_c6(positions, box, pairs, mScales, a_list, b_list, q_list, c6_list)
    E4 = onebodyenergy(positions, box)
    
    return E1 - E2 + E3 + E4

