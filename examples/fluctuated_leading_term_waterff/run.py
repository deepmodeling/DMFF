#!/usr/bin/env python
import sys
from pathlib import Path
admp_path = Path(__file__).parent.parent.parent
sys.path.append(str(admp_path))
import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
from jax_md import partition, space
#from dmff.admp.multipole import convert_cart2harm
from dmff.admp.pme import ADMPPmeForce,trim_val_0
from dmff.admp.disp_pme import ADMPDispPmeForce
from dmff.admp.pairwise import generate_pairwise_interaction, TT_damping_qq_c6_kernel
from dmff.admp.parser import *
from dmff.utils import jit_condition
from dmff.admp.spatial import v_pbc_shift

#compute the fluctuated leading term using the linear model  
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
    dipole = -0.016858755+0.002287251*angle + 0.239667591*dROH1 + (-0.070483437)*dROH2
    charge_H = dipole/dROH1
    charge_O=charge_H*(-2)
    C6_H = (-2.36066199 + (-0.007049238)*angle + 1.949429648*dROH1+ 2.097120784*dROH2) * 0.529**6 * 2625.5
    C6_O = (-8.641301261 + 0.093247893*angle + 11.90395358*(dROH1+ dROH2)) * 0.529**6 * 2625.5
    C6_H = trim_val_0(C6_H)
    c0 = c0.at[::3].set(charge_O)
    c0 = c0.at[1::3].set(charge_H)
    c0 = c0.at[2::3].set(charge_H)
    c6_list = c6_list.at[::3].set(jnp.sqrt(C6_O))
    c6_list = c6_list.at[1::3].set(jnp.sqrt(C6_H))
    c6_list = c6_list.at[2::3].set(jnp.sqrt(C6_H))
    return c0, c6_list

#compute energy with fluctuated leading terms
def admp_calculator(positions, Q_local, axis_type, axis_indices, box, covalent_map, pol, tholes, c_list, q_list, a_list, b_list, pairs, rc):
    c0, c6_list = compute_leading_terms(positions, box)
    Q_local = Q_local.at[:,0].set(c0)  #set fluctuated charge
    c_list = c_list.at[:,0].set(c6_list) #set fluctuated C6
    q_list = c0
    
    E1 = pme_force.get_energy(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales)
    E2 = disp_pme_force.get_energy(positions, box, pairs, c_list, mScales)
    E3 = TT_damping_qq_c6(positions, box, pairs, mScales, a_list, b_list, q_list, c_list[:,0])
    return E1 - E2 + E3 


# below is the validation code
if __name__ == '__main__':
    pdb = 'water_dimer.pdb'
    xml = 'forcefield.xml'

    positions, Q_local, axis_type, axis_indices, box, covalent_map, pol, tholes, c_list, q_list, a_list, b_list \
            = read_input_info(pdb, xml)

    # setting of cutoff and scale factors
    rc = 4
    ethresh = 1e-4
    mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    lmax = 2
    pmax = 10
    lpol = True # the polarization will turned on if True


    pme_force = ADMPPmeForce(box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax, lpol, steps_pol=5)
    disp_pme_force = ADMPDispPmeForce(box, covalent_map, rc, ethresh, pmax)
    TT_damping_qq_c6 = generate_pairwise_interaction(TT_damping_qq_c6_kernel, covalent_map, static_args={})

    #compute neighbour list
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T

    tot_force = value_and_grad(admp_calculator,argnums=(0))
    ene, force = tot_force(positions, Q_local, axis_type, axis_indices, box, covalent_map, pol, tholes, c_list, q_list, a_list, b_list, pairs, rc)
    print(ene)
    print(force)
