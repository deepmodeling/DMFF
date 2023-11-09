import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from dmff.api import DMFFTopology
from jax import jit, value_and_grad

class TestADMPSlaterTypeFunction:

    @pytest.mark.parametrize(
        "pdb, prm, values", 
        [
            (
                "tests/data/peg2.pdb", 
                "tests/data/peg.xml", 
                # "peg2.pdb", 
                # "peg.xml", 
                jnp.array([
                    5.731787, -0.053504415, 4.510135e-09, -4.510135e-09, -4.510135e-09,
                    -4.510135e-09, -4.510135e-09, -5.505022e-11, 1.0166141e-06
                ])
            ),
        ]
    )
    def test_admp_slater(self, pdb, prm, values):
        pdb_AB = app.PDBFile(pdb)
        H_AB = Hamiltonian(prm)
        rc = 1.49
        pots_AB = H_AB.createPotential(
            pdb_AB.topology, 
            nonbondedCutoff=rc*unit.nanometer, 
            nonbondedMethod=app.CutoffPeriodic, 
            ethresh=1e-4)

        pot_pme_AB = pots_AB.dmff_potentials['ADMPPmeForce']
        pot_disp_AB = pots_AB.dmff_potentials['ADMPDispPmeForce']
        pot_ex_AB = pots_AB.dmff_potentials['SlaterExForce']
        pot_sr_es_AB = pots_AB.dmff_potentials['SlaterSrEsForce']
        pot_sr_pol_AB = pots_AB.dmff_potentials['SlaterSrPolForce']
        pot_sr_disp_AB = pots_AB.dmff_potentials['SlaterSrDispForce']
        pot_dhf_AB = pots_AB.dmff_potentials['SlaterDhfForce']
        pot_dmp_es_AB = pots_AB.dmff_potentials['QqTtDampingForce']
        pot_dmp_disp_AB = pots_AB.dmff_potentials['SlaterDampingForce']

        # build cov map
        dmfftop = DMFFTopology(from_top=pdb_AB.topology)
        covalent_map = dmfftop.buildCovMat()

        # init positions used to set up neighbor list
        pos_AB0 = jnp.array(pdb_AB.positions._value)
        n_atoms = len(pos_AB0)
        box = jnp.array(pdb_AB.topology.getPeriodicBoxVectors()._value)

        # nn list initial allocation
        nbl_AB = NeighborList(box, rc, covalent_map)
        nbl_AB.allocate(pos_AB0)
        pairs_AB = nbl_AB.pairs
        pairs_AB =  pairs_AB[pairs_AB[:, 0] < pairs_AB[:, 1]]

        pos_AB = jnp.array(pos_AB0)
        E_es = pot_pme_AB(pos_AB, box, pairs_AB, H_AB.paramset)
        E_disp = pot_disp_AB(pos_AB, box, pairs_AB, H_AB.paramset)
        E_ex = pot_ex_AB(pos_AB, box, pairs_AB, H_AB.paramset)
        E_sr_es = pot_sr_es_AB(pos_AB, box, pairs_AB, H_AB.paramset) 
        E_sr_pol = pot_sr_pol_AB(pos_AB, box, pairs_AB, H_AB.paramset)
        E_sr_disp = pot_sr_disp_AB(pos_AB, box, pairs_AB, H_AB.paramset) 
        E_dhf = pot_dhf_AB(pos_AB, box, pairs_AB, H_AB.paramset)
        E_dmp_es = pot_dmp_es_AB(pos_AB, box, pairs_AB, H_AB.paramset) 
        E_dmp_disp = pot_dmp_disp_AB(pos_AB, box, pairs_AB, H_AB.paramset)

        npt.assert_almost_equal(E_es, values[0], decimal=4)
        npt.assert_almost_equal(E_disp, values[1], decimal=4)
        npt.assert_almost_equal(E_ex, values[2], decimal=4)
        npt.assert_almost_equal(E_sr_es, values[3], decimal=4)
        npt.assert_almost_equal(E_sr_pol, values[4], decimal=4)
        npt.assert_almost_equal(E_sr_disp, values[5], decimal=4) 
        npt.assert_almost_equal(E_dhf, values[6], decimal=4)
        npt.assert_almost_equal(E_dmp_es, values[7], decimal=4)
        npt.assert_almost_equal(E_dmp_disp, values[8], decimal=4)
