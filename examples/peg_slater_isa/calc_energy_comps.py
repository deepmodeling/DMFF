#!/usr/bin/env python
import numpy as np
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import jax
import jax_md
import jax.numpy as jnp
import dmff
from dmff.api import Hamiltonian
import pickle
import time


if __name__ == '__main__':
    ff = 'forcefield.xml'
    pdb_AB = PDBFile('peg2_dimer.pdb')
    pdb_A = PDBFile('peg2.pdb')
    pdb_B = PDBFile('peg2.pdb')
    H_AB = Hamiltonian(ff)
    H_A = Hamiltonian(ff)
    H_B = Hamiltonian(ff)
    pme_generator_AB, \
            disp_generator_AB, \
            ex_generator_AB, \
            sr_es_generator_AB, \
            sr_pol_generator_AB, \
            sr_disp_generator_AB, \
            dhf_generator_AB, \
            dmp_es_generator_AB, \
            dmp_disp_generator_AB = H_AB.getGenerators()
    pme_generator_A, \
            disp_generator_A, \
            ex_generator_A, \
            sr_es_generator_A, \
            sr_pol_generator_A, \
            sr_disp_generator_A, \
            dhf_generator_A, \
            dmp_es_generator_A, \
            dmp_disp_generator_A = H_A.getGenerators()
    pme_generator_B, \
            disp_generator_B, \
            ex_generator_B, \
            sr_es_generator_B, \
            sr_pol_generator_B, \
            sr_disp_generator_B, \
            dhf_generator_B, \
            dmp_es_generator_B, \
            dmp_disp_generator_B = H_B.getGenerators()

    rc = 15

    # get potential functions
    potentials_AB = H_AB.createPotential(pdb_AB.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_AB, \
            pot_disp_AB, \
            pot_ex_AB, \
            pot_sr_es_AB, \
            pot_sr_pol_AB, \
            pot_sr_disp_AB, \
            pot_dhf_AB, \
            pot_dmp_es_AB, \
            pot_dmp_disp_AB = potentials_AB
    potentials_A = H_A.createPotential(pdb_A.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_A, \
            pot_disp_A, \
            pot_ex_A, \
            pot_sr_es_A, \
            pot_sr_pol_A, \
            pot_sr_disp_A, \
            pot_dhf_A, \
            pot_dmp_es_A, \
            pot_dmp_disp_A = potentials_A
    potentials_B = H_B.createPotential(pdb_B.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_B, \
            pot_disp_B, \
            pot_ex_B, \
            pot_sr_es_B, \
            pot_sr_pol_B, \
            pot_sr_disp_B, \
            pot_dhf_B, \
            pot_dmp_es_B, \
            pot_dmp_disp_B = potentials_B

    pos_AB0 = jnp.array(pdb_AB.positions._value) * 10
    n_atoms = len(pos_AB0)
    n_atoms_A = n_atoms // 2
    n_atoms_B = n_atoms // 2
    pos_A0 = jnp.array(pdb_AB.positions._value[:n_atoms_A]) * 10
    pos_B0 = jnp.array(pdb_AB.positions._value[n_atoms_A:n_atoms]) * 10
    box = jnp.array(pdb_AB.topology.getPeriodicBoxVectors()._value) * 10
    # nn list initial allocation
    displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = jax_md.partition.neighbor_list(displacement_fn, box, rc, 0, format=jax_md.partition.OrderedSparse)
    nbr_AB = neighbor_list_fn.allocate(pos_AB0)
    nbr_A = neighbor_list_fn.allocate(pos_A0)
    nbr_B = neighbor_list_fn.allocate(pos_B0)
    pairs_AB = np.array(nbr_AB.idx.T)
    pairs_A = np.array(nbr_A.idx.T)
    pairs_B = np.array(nbr_B.idx.T)
    pairs_AB =  pairs_AB[pairs_AB[:, 0] < pairs_AB[:, 1]]
    pairs_A =  pairs_A[pairs_A[:, 0] < pairs_A[:, 1]]
    pairs_B =  pairs_B[pairs_B[:, 0] < pairs_B[:, 1]]

    # load data
    with open('data.pickle', 'rb') as ifile:
        data = pickle.load(ifile)

    keys = list(data.keys())
    keys.sort()
    # for sid in keys:
    for sid in ['000']:
        scan_res = data[sid]
        scan_res['tot_full'] = scan_res['tot'].copy()
        npts = len(scan_res['tot'])
        print(sid)

        for ipt in range(npts):
            E_es_ref = scan_res['es'][ipt]
            E_pol_ref = scan_res['pol'][ipt]
            E_disp_ref = scan_res['disp'][ipt]
            E_ex_ref = scan_res['ex'][ipt]
            E_dhf_ref = scan_res['dhf'][ipt]
            E_tot_ref = scan_res['tot'][ipt]

            # get position array
            pos_A = jnp.array(scan_res['posA'][ipt])
            pos_B = jnp.array(scan_res['posB'][ipt])
            pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)
     

            #####################
            # exchange repulsion
            #####################
            E_ex_AB = pot_ex_AB(pos_AB, box, pairs_AB, ex_generator_AB.params)
            E_ex_A = pot_ex_A(pos_A, box, pairs_A, ex_generator_AB.params)
            E_ex_B = pot_ex_B(pos_B, box, pairs_B, ex_generator_AB.params)
            E_ex = E_ex_AB - E_ex_A - E_ex_B

            #######################
            # electrostatic + pol
            #######################
            E_AB = pot_pme_AB(pos_AB, box, pairs_AB, pme_generator_AB.params)
            E_A = pot_pme_A(pos_A, box, pairs_A, pme_generator_A.params)
            E_B = pot_pme_B(pos_B, box, pairs_A, pme_generator_B.params)
            E_espol = E_AB - E_A - E_B

            # use induced dipole of monomers to compute electrostatic interaction
            U_ind_AB = jnp.vstack((pme_generator_A.pme_force.U_ind, pme_generator_B.pme_force.U_ind))
            params = pme_generator_AB.params
            map_atypes = pme_generator_AB.map_atomtype
            Q_local = params['Q_local'][map_atypes]
            pol = params['pol'][map_atypes]
            tholes = params['tholes'][map_atypes]
            pme_force = pme_generator_AB.pme_force
            E_AB_nonpol = pme_force.energy_fn(pos_AB, box, pairs_AB, Q_local, U_ind_AB, pol, tholes, params['mScales'], params['pScales'], params['dScales'])
            E_es = E_AB_nonpol - E_A - E_B
            E_dmp_es = pot_dmp_es_AB(pos_AB, box, pairs_AB, dmp_es_generator_AB.params) \
                     - pot_dmp_es_A(pos_A, box, pairs_A, dmp_es_generator_A.params) \
                     - pot_dmp_es_B(pos_B, box, pairs_B, dmp_es_generator_B.params)
            E_sr_es = pot_sr_es_AB(pos_AB, box, pairs_AB, sr_es_generator_AB.params) \
                    - pot_sr_es_A(pos_A, box, pairs_A, sr_es_generator_AB.params) \
                    - pot_sr_es_B(pos_B, box, pairs_B, sr_es_generator_AB.params)


            ###################################
            # polarization (induction) energy
            ###################################
            E_pol = E_espol - E_es
            E_sr_pol = pot_sr_pol_AB(pos_AB, box, pairs_AB, sr_pol_generator_AB.params) \
                     - pot_sr_pol_A(pos_A, box, pairs_A, sr_pol_generator_AB.params) \
                     - pot_sr_pol_B(pos_B, box, pairs_B, sr_pol_generator_AB.params)


            #############
            # dispersion
            #############
            E_AB_disp = pot_disp_AB(pos_AB, box, pairs_AB, disp_generator_AB.params)
            E_A_disp = pot_disp_A(pos_A, box, pairs_A, disp_generator_AB.params)
            E_B_disp = pot_disp_B(pos_B, box, pairs_B, disp_generator_AB.params)
            E_disp = E_AB_disp - E_A_disp - E_B_disp
            E_dmp_disp = pot_dmp_disp_AB(pos_AB, box, pairs_AB, dmp_disp_generator_AB.params) \
                       - pot_dmp_disp_A(pos_A, box, pairs_A, dmp_disp_generator_A.params) \
                       - pot_dmp_disp_B(pos_B, box, pairs_B, dmp_disp_generator_B.params)
            E_sr_disp = pot_sr_disp_AB(pos_AB, box, pairs_AB, sr_disp_generator_AB.params) \
                      - pot_sr_disp_A(pos_A, box, pairs_A, sr_disp_generator_AB.params) \
                      - pot_sr_disp_B(pos_B, box, pairs_B, sr_disp_generator_AB.params)

            ###########
            # dhf
            ###########
            E_AB_dhf = pot_dhf_AB(pos_AB, box, pairs_AB, dhf_generator_AB.params)
            E_A_dhf = pot_dhf_A(pos_A, box, pairs_A, dhf_generator_AB.params)
            E_B_dhf = pot_dhf_B(pos_B, box, pairs_B, dhf_generator_AB.params)
            E_dhf = E_AB_dhf - E_A_dhf - E_B_dhf

            # total energy
            E_tot = (E_es + E_sr_es + E_dmp_es) + (E_ex) + (E_pol + E_sr_pol) + (E_disp + E_dmp_disp + E_sr_disp) + (E_dhf)
            # print(E_dmp_es + E_disp + E_dmp_disp)
            print(E_es + E_pol)

