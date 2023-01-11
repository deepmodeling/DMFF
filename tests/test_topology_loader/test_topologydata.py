import pytest
import openmm.app as app
import openmm.unit as unit
from dmff.topology import TopologyData
from dmff.template.ffxml import generate_templates_from_xml


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
        "pdb, ref",
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

    @pytest.mark.parametrize(
        "pdb, ref",
        [
            ("tests/data/lig.pdb", "test/data/lig.mol2")
        ]
    )
    def test_match_atomtype(self, pdb, ref):
        # load
        pdb_omm = app.PDBFile(pdb)
        topo = pdb_omm.topology
        # use TopologyData
        data = TopologyData(topo)
        templates = generate_templates_from_xml("tests/data/gaff-2.11.xml", "tests/data/lig-prm-full.xml")
        data.match_all(templates)
        atom_types = [d['type'] for d in data.properties]
        pass

    