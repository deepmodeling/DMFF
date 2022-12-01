#!/usr/bin/env python3
import os
import sys
import driver
import numpy as np
import jax
import jax.numpy as jnp
import dmff.admp.pme
import openmm.app as app
from model6 import compute_leading_terms
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.common import nblist
from dmff.admp.parser import *
from dmff.admp.pairwise import (
    generate_pairwise_interaction,
    TT_damping_qq_kernel
)
from intra import onebodyenergy
from jax_md import space, partition
from jax import jit, vmap
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

dmff.admp.pme.DEFAULT_THOLE_WIDTH = 2.6

@vmap
@jit
def TT_damping_qq_disp_kernel(dr, m, ai, aj, bi, bj, qi, qj, c6i, c6j, c8i, c8j, c10i, c10j):
    a = jnp.sqrt(ai * aj)
    b = jnp.sqrt(bi * bj)
    c6 = c6i * c6j
    c8 = c8i * c8j
    c10 = c10i * c10j
    q = qi * qj
    r = dr * 1.889726878 # convert to bohr
    br = b * r
    br2 = br * br
    br3 = br2 * br
    br4 = br2 * br2
    br5 = br3 * br2
    br6 = br3 * br3
    br7 = br3 * br4
    br8 = br4 * br4
    br9 = br4 * br5
    br10 = br5 * br5
    exp_br = jnp.exp(-br)
    f = 2625.5 * a * exp_br \
        + (-2625.5) * exp_br * (1+br) * q / r \
        + exp_br*(1+br+br2/2+br3/6+br4/24+br5/120+br6/720) * c6 / dr**6 \
        + exp_br*(1+br+br2/2+br3/6+br4/24+br5/120+br6/720+br7/5040+br8/40320) * c8 / dr**8 \
        + exp_br*(1+br+br2/2+br3/6+br4/24+br5/120+br6/720+br7/5040+br8/40320+br9/362880+br10/3628800) * c10 / dr**10

    return f * m

class DMFFDriver(driver.BaseDriver):

    def __init__(self, addr, port, pdb, ffxml, topo_xml, socktype, device='cpu'):
        addr = addr + '_%s' %os.environ['SLURM_JOB_ID']
        # set up the interface with ipi
        driver.BaseDriver.__init__(self, port, addr, socktype)
        # set up various force calculators
        #self.pdb = pdb
        H = Hamiltonian(ffxml)
        app.Topology.loadBondDefinitions(topo_xml)
        pdb = app.PDBFile(pdb)
        positions = jnp.array(pdb.positions._value) * 10
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10 
        #self.box = box
        rc = 4
        disp_generator, pme_generator = H.getGenerators()
        pots = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom, step_pol=5)
        pot_disp = pots.dmff_potentials['ADMPDispForce']
        pot_pme = pots.dmff_potentials['ADMPPmeForce']
        pot_sr = generate_pairwise_interaction(TT_damping_qq_disp_kernel, static_args={})


        self.nbl = nblist.NeighborList(box, rc, H.getGenerators()[0].covalent_map)
        self.nbl.allocate(positions)
        pairs = self.nbl.pairs
      
        def admp_calculator(positions, box, pairs):
            params_pme = pme_generator.paramtree['ADMPPmeForce']
            params_disp = disp_generator.paramtree['ADMPDispForce']
            c0, c6_list = compute_leading_terms(positions,box) # compute fluctuated leading terms
            Q_local = params_pme["Q_local"][pme_generator.map_atomtype]
            Q_local = Q_local.at[:,0].set(c0)  # change fixed charge into fluctuated one
            pol = params_pme["pol"][pme_generator.map_atomtype]
            tholes = params_pme["tholes"][pme_generator.map_atomtype]
            c8_list = jnp.sqrt(params_disp["C8"][disp_generator.map_atomtype]*1e8)
            c10_list = jnp.sqrt(params_disp["C10"][disp_generator.map_atomtype]*1e10)
            c_list = jnp.vstack((c6_list, c8_list, c10_list))
            covalent_map = disp_generator.covalent_map
            a_list = (params_disp["A"][disp_generator.map_atomtype] / 2625.5)
            b_list = params_disp["B"][disp_generator.map_atomtype] * 0.0529177249
            q_list = c0
            E_pme = pme_generator.pme_force.get_energy(
                    positions, box, pairs, Q_local, pol, tholes, params_pme["mScales"], params_pme["pScales"], params_pme["dScales"]
                    )
            E_disp = disp_generator.disp_pme_force.get_energy(positions, box, pairs, c_list.T, params_disp["mScales"])
            E_sr = pot_sr(positions, box, pairs, params_disp["mScales"], a_list, b_list, q_list, c_list[0], c_list[1], c_list[2])            

            E4 = onebodyenergy(positions, box)
            
            return E_pme - E_disp + E_sr + E4
        self.tot_force = jit(jax.value_and_grad(admp_calculator,argnums=(0)))

        # compile tot_force function
        E, F = self.tot_force(positions, box, pairs)


    def grad(self, crd, cell): # receive SI input, return SI values
        positions = jnp.array(crd*1e10) # convert to angstrom
        box = jnp.array(cell*1e10)      # convert to angstrom
        # nb list
        nbl = self.nbl.allocate(positions)
        pairs = self.nbl.pairs
        energy, grad = self.tot_force(positions, box, pairs)
        energy = np.float64(energy)
        grad = np.float64(grad)
        energy = energy * 1000 / 6.0221409e+23 # kj/mol to Joules
        grad = grad * 1000 / 6.0221409e+23 * 1e10 # convert kj/mol/A to joule/m

        return energy, grad


if __name__ == '__main__':
    # the forces are composed by three parts: 
    # the long range part computed using openmm, parameters in xml
    # the short range part writen by hand, parameters in psr
    fn_pdb = sys.argv[1] # pdb file used to define openmm topology, this one should contain all virtual sites
    ff_xml = sys.argv[2] # xml file that defines the force field
    topo_xml = sys.argv[3]
    addr = sys.argv[4]
    port = int(sys.argv[5])
    socktype = sys.argv[6] 
    driver_dmff = DMFFDriver(addr, port, fn_pdb, ff_xml, topo_xml, socktype)
    while True:
        driver_dmff.parse()
