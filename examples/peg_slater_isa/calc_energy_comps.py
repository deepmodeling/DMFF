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
from dmff.common import nblist
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

    rc = 1.45

    # get potential functions
    pots_AB = H_AB.createPotential(pdb_AB.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_AB = pots_AB.dmff_potentials['ADMPPmeForce']
    pot_disp_AB = pots_AB.dmff_potentials['ADMPDispPmeForce']
    pot_ex_AB = pots_AB.dmff_potentials['SlaterExForce']
    pot_sr_es_AB = pots_AB.dmff_potentials['SlaterSrEsForce']
    pot_sr_pol_AB = pots_AB.dmff_potentials['SlaterSrPolForce']
    pot_sr_disp_AB = pots_AB.dmff_potentials['SlaterSrDispForce']
    pot_dhf_AB = pots_AB.dmff_potentials['SlaterDhfForce']
    pot_dmp_es_AB = pots_AB.dmff_potentials['QqTtDampingForce']
    pot_dmp_disp_AB = pots_AB.dmff_potentials['SlaterDampingForce']
    pots_A = H_A.createPotential(pdb_A.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_A = pots_A.dmff_potentials['ADMPPmeForce']
    pot_disp_A = pots_A.dmff_potentials['ADMPDispPmeForce']
    pot_ex_A = pots_A.dmff_potentials['SlaterExForce']
    pot_sr_es_A = pots_A.dmff_potentials['SlaterSrEsForce']
    pot_sr_pol_A = pots_A.dmff_potentials['SlaterSrPolForce']
    pot_sr_disp_A = pots_A.dmff_potentials['SlaterSrDispForce']
    pot_dhf_A = pots_A.dmff_potentials['SlaterDhfForce']
    pot_dmp_es_A = pots_A.dmff_potentials['QqTtDampingForce']
    pot_dmp_disp_A = pots_A.dmff_potentials['SlaterDampingForce']
    pots_B = H_B.createPotential(pdb_B.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_B = pots_B.dmff_potentials['ADMPPmeForce']
    pot_disp_B = pots_B.dmff_potentials['ADMPDispPmeForce']
    pot_ex_B = pots_B.dmff_potentials['SlaterExForce']
    pot_sr_es_B = pots_B.dmff_potentials['SlaterSrEsForce']
    pot_sr_pol_B = pots_B.dmff_potentials['SlaterSrPolForce']
    pot_sr_disp_B = pots_B.dmff_potentials['SlaterSrDispForce']
    pot_dhf_B = pots_B.dmff_potentials['SlaterDhfForce']
    pot_dmp_es_B = pots_B.dmff_potentials['QqTtDampingForce']
    pot_dmp_disp_B = pots_B.dmff_potentials['SlaterDampingForce']
    pme_generator_AB = H_AB.getGenerators()[0]
    pme_generator_A = H_A.getGenerators()[0]
    pme_generator_B = H_B.getGenerators()[0]

    pos_AB0 = jnp.array(pdb_AB.positions._value)
    n_atoms = len(pos_AB0)
    n_atoms_A = n_atoms // 2
    n_atoms_B = n_atoms // 2
    pos_A0 = jnp.array(pdb_AB.positions._value[:n_atoms_A])
    pos_B0 = jnp.array(pdb_AB.positions._value[n_atoms_A:n_atoms])
    box = jnp.array(pdb_AB.topology.getPeriodicBoxVectors()._value)
    # nn list initial allocation
    nbl_AB = nblist.NeighborList(box, rc, pots_AB.meta["cov_map"])
    nbl_AB.allocate(pos_AB0)
    pairs_AB = nbl_AB.pairs
    nbl_A = nblist.NeighborList(box, rc, pots_A.meta["cov_map"])
    nbl_A.allocate(pos_A0)
    pairs_A = nbl_A.pairs
    nbl_B = nblist.NeighborList(box, rc, pots_B.meta["cov_map"])
    nbl_B.allocate(pos_B0)
    pairs_B = nbl_B.pairs

    pairs_AB =  pairs_AB[pairs_AB[:, 0] < pairs_AB[:, 1]]
    pairs_A =  pairs_A[pairs_A[:, 0] < pairs_A[:, 1]]
    pairs_B =  pairs_B[pairs_B[:, 0] < pairs_B[:, 1]]


    params = H_AB.getParameters()

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
            pos_A = jnp.array(scan_res['posA'][ipt]) / 10
            pos_B = jnp.array(scan_res['posB'][ipt]) / 10
            pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)
     

            #####################
            # exchange repulsion
            #####################
            E_ex_AB = pot_ex_AB(pos_AB, box, pairs_AB, params)
            E_ex_A = pot_ex_A(pos_A, box, pairs_A, params)
            E_ex_B = pot_ex_B(pos_B, box, pairs_B, params)
            E_ex = E_ex_AB - E_ex_A - E_ex_B

            #######################
            # electrostatic + pol
            #######################
            E_AB = pot_pme_AB(pos_AB, box, pairs_AB, params)
            E_A = pot_pme_A(pos_A, box, pairs_A, params)
            E_B = pot_pme_B(pos_B, box, pairs_A, params)
            E_espol = E_AB - E_A - E_B

            # use induced dipole of monomers to compute electrostatic interaction
            U_ind_AB = jnp.vstack((pme_generator_A.pme_force.U_ind, pme_generator_B.pme_force.U_ind))
            params_pme = params['ADMPPmeForce']
            map_atypes = pots_AB.meta['ADMPPmeForce_map_atomtype']
            map_poltype = pots_AB.meta['ADMPPmeForce_map_poltype']
            Q_local = params_pme['Q_local'][map_atypes]
            pol = params_pme['pol'][map_poltypes]
            tholes = params_pme['thole'][map_poltypes]
            pme_force = pme_generator_AB.pme_force


            E_AB_nonpol = pme_force.energy_fn(pos_AB*10, box*10, pairs_AB, Q_local, U_ind_AB, pol, tholes, \
                    pme_generator_AB.mScales, pme_generator_AB.pScales, pme_generator_AB.dScales)
            E_es = E_AB_nonpol - E_A - E_B


            E_dmp_es = pot_dmp_es_AB(pos_AB, box, pairs_AB, params) \
                     - pot_dmp_es_A(pos_A, box, pairs_A, params) \
                     - pot_dmp_es_B(pos_B, box, pairs_B, params)
            E_sr_es = pot_sr_es_AB(pos_AB, box, pairs_AB, params) \
                    - pot_sr_es_A(pos_A, box, pairs_A, params) \
                    - pot_sr_es_B(pos_B, box, pairs_B, params)


            ###################################
            # polarization (induction) energy
            ###################################
            E_pol = E_espol - E_es
            E_sr_pol = pot_sr_pol_AB(pos_AB, box, pairs_AB, params) \
                     - pot_sr_pol_A(pos_A, box, pairs_A, params) \
                     - pot_sr_pol_B(pos_B, box, pairs_B, params)


            #############
            # dispersion
            #############
            E_AB_disp = pot_disp_AB(pos_AB, box, pairs_AB, params)
            E_A_disp = pot_disp_A(pos_A, box, pairs_A, params)
            E_B_disp = pot_disp_B(pos_B, box, pairs_B, params)
            E_disp = E_AB_disp - E_A_disp - E_B_disp
            E_dmp_disp = pot_dmp_disp_AB(pos_AB, box, pairs_AB, params) \
                       - pot_dmp_disp_A(pos_A, box, pairs_A, params) \
                       - pot_dmp_disp_B(pos_B, box, pairs_B, params)
            E_sr_disp = pot_sr_disp_AB(pos_AB, box, pairs_AB, params) \
                      - pot_sr_disp_A(pos_A, box, pairs_A, params) \
                      - pot_sr_disp_B(pos_B, box, pairs_B, params)

            ###########
            # dhf
            ###########
            E_AB_dhf = pot_dhf_AB(pos_AB, box, pairs_AB, params)
            E_A_dhf = pot_dhf_A(pos_A, box, pairs_A, params)
            E_B_dhf = pot_dhf_B(pos_B, box, pairs_B, params)
            E_dhf = E_AB_dhf - E_A_dhf - E_B_dhf

            # total energy
            E_tot = (E_es + E_sr_es + E_dmp_es) + (E_ex) + (E_pol + E_sr_pol) + (E_disp + E_dmp_disp + E_sr_disp) + (E_dhf)
            # print(E_dmp_es + E_disp + E_dmp_disp)
            print(E_es + E_pol)

