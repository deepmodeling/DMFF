import pytest
import openmm.app as app
import openmm.unit as unit
from dmff.topology import TopologyData, is_propers_equal, is_improper_equal


class TestTopologyData:

    @pytest.mark.parametrize(
        "pdb",
        [
            ("tests/data/lig.pdb")
        ]
    )
    def test_load_topology(self, pdb):
        # load
        pdb_omm = app.PDBFile(pdb)
        topo = pdb_omm.topology
        # generate reference
        pass
        # use TopologyData
        data = TopologyData(topo)
        # check consistency
        pass

    @pytest.mark.parametrize(
        "pdb",
        [
            ("tests/data/lig.pdb", "test/data/lig_impr.txt")
        ]
    )
    def test_detect_improper(self, pdb):
        # load
        pdb_omm = app.PDBFile(pdb)
        topo = pdb_omm.topology
        # use TopologyData
        data = TopologyData(topo)
        data.detect_impropers()
        for pair in data.impropers:
            pass