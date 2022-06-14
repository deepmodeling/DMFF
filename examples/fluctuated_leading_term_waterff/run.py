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

def generate_calculator(pot_disp, pot_pme, pot_sr, disp_generator, pme_generator):
    def admp_calculator(positions, box, pairs):
        c0, c6_list = compute_leading_terms(positions,box) # compute fluctuated leading terms
        Q_local = pme_generator.params["Q_local"][pme_generator.map_atomtype]
        Q_local = Q_local.at[:,0].set(c0)  # change fixed charge into fluctuated one
        pol=pme_generator.params["pol"][pme_generator.map_atomtype]
        tholes=pme_generator.params["tholes"][pme_generator.map_atomtype]
        c8_list = jnp.sqrt(disp_generator.params["C8"][disp_generator.map_atomtype]*1e8)
        c10_list = jnp.sqrt(disp_generator.params["C10"][disp_generator.map_atomtype]*1e10)
        c_list = jnp.vstack((c6_list, c8_list, c10_list))
        covalent_map = disp_generator.disp_pme_force.covalent_map    
        a_list = (disp_generator.params["A"][disp_generator.map_atomtype] / 2625.5)
        b_list=disp_generator.params["B"][disp_generator.map_atomtype] * 0.0529177249 
        q_list = disp_generator.params["Q"][disp_generator.map_atomtype]

        E_pme = pme_generator.pme_force.get_energy(
                positions, box, pairs, Q_local, pol, tholes, pme_generator.params["mScales"], pme_generator.params["pScales"], pme_generator.params["dScales"]
                )    
        E_disp = disp_generator.disp_pme_force.get_energy(positions, box, pairs, c_list.T, disp_generator.params["mScales"])
        E_sr = pot_sr(positions, box, pairs, disp_generator.params["mScales"], a_list, b_list, q_list, c_list[0])
        return E_pme + E_sr - E_disp
    return jit(value_and_grad(admp_calculator,argnums=(0)))

if __name__ == '__main__':
    
    H = Hamiltonian('forcefield.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("water_dimer.pdb")
    rc = 4

    # generator stores all force field parameters     
    disp_generator, pme_generator = H.getGenerators()
    pot_disp, pot_pme = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom, step_pol=5)
    pot_sr = generate_pairwise_interaction(TT_damping_qq_c6_kernel, disp_generator.disp_pme_force.covalent_map, static_args={})
    
    # construct inputs
    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10
    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T    

    
    admp_calc = generate_calculator(pot_disp, pot_pme, pot_sr, disp_generator, pme_generator)
    tot_ene, tot_force = admp_calc(positions, box, pairs)
    print('# Tot Interaction Energy:')
    print('#', tot_ene, 'kJ/mol')
    print('# Tot force :')
    print('#', tot_force)
