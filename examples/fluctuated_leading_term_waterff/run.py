#!/usr/bin/env python
import sys
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from jax_md import space, partition
from jax import value_and_grad, jit
import pickle
from dmff.admp.pme import trim_val_0
from dmff.admp.spatial import v_pbc_shift
from dmff.common import nblist
from dmff.utils import jit_condition
from dmff.admp.pairwise import (
    TT_damping_qq_c6_kernel,
    generate_pairwise_interaction,
    slater_disp_damping_kernel,
    slater_sr_kernel,
    TT_damping_qq_kernel
)

@jit_condition(static_argnums=())
def compute_leading_terms(positions,box):
    n_atoms = len(positions)
    c0 = jnp.zeros(n_atoms)
    c6_list = jnp.zeros(n_atoms)
    box_inv = jnp.linalg.inv(box + jnp.eye(3) * 1e-36)
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

# def generate_calculator(pot_disp, pot_pme, pot_sr, disp_generator, pme_generator, params, covalent_map):
def generate_calculator(pots, pme_generator, disp_generator, params):
    pot_disp = pots.dmff_potentials['ADMPDispForce']
    pot_pme = pots.dmff_potentials['ADMPPmeForce']
    pot_sr = generate_pairwise_interaction(TT_damping_qq_c6_kernel, static_args={})
    map_atype_disp = pots.meta["ADMPDispForce_map_atomtype"]
    map_atype_pme = pots.meta["ADMPPmeForce_map_atomtype"]
    def admp_calculator(positions, box, pairs):
        # params_pme = pme_generator.paramtree['ADMPPmeForce']
        # params_disp = disp_generator.paramtree['ADMPDispForce']
        params_pme = params['ADMPPmeForce']
        params_disp = params['ADMPDispForce']
        c0, c6_list = compute_leading_terms(positions,box) # compute fluctuated leading terms
        Q_local = params_pme["Q_local"][map_atype_pme]
        Q_local = Q_local.at[:,0].set(c0)  # change fixed charge into fluctuated one
        pol = params_pme["pol"][map_atype_pme]
        tholes = params_pme["thole"][map_atype_pme]
        c8_list = jnp.sqrt(params_disp["C8"][map_atype_disp]*1e8)
        c10_list = jnp.sqrt(params_disp["C10"][map_atype_disp]*1e10)
        c_list = jnp.vstack((c6_list, c8_list, c10_list))
        a_list = (params_disp["A"][map_atype_disp] / 2625.5)
        b_list = params_disp["B"][map_atype_disp] * 0.0529177249 
        q_list = params_disp["Q"][map_atype_disp]

        E_pme = pme_generator.pme_force.get_energy(
                positions, box, pairs, Q_local, pol, tholes, pme_generator.mScales, pme_generator.pScales, pme_generator.dScales
                )    
        E_disp = disp_generator.disp_pme_force.get_energy(positions, box, pairs, c_list.T, disp_generator.mScales)
        E_sr = pot_sr(positions, box, pairs, disp_generator.mScales, a_list, b_list, q_list, c_list[0])
        return E_pme 
    return jit(value_and_grad(admp_calculator,argnums=(0)))

if __name__ == '__main__':
    
    H = Hamiltonian('forcefield.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("water_dimer.pdb")
    rc = 0.4

    # generator stores all force field parameters     
    disp_generator, pme_generator = H.getGenerators()
    pots = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.nanometer, step_pol=5)
    # pot_disp = pots.dmff_potentials['ADMPDispForce']
    # pot_pme = pots.dmff_potentials['ADMPPmeForce']
    # pot_sr = generate_pairwise_interaction(TT_damping_qq_c6_kernel, static_args={})
    
    # construct inputs
    positions = jnp.array(pdb.positions._value)
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value])
    # neighbor list
    nbl = nblist.NeighborList(box, rc, pots.meta["cov_map"])
    nbl.allocate(positions)
    pairs = nbl.pairs

    
    admp_calc = generate_calculator(pots, pme_generator, disp_generator, H.getParameters())
    tot_ene, tot_force = admp_calc(positions*10, box*10, pairs)
    print('# Tot Interaction Energy:')
    print('#', tot_ene, 'kJ/mol')
    print('# Tot force :')
    print('#', tot_force)
