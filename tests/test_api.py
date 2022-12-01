import subprocess
from pathlib import Path
import openmm.app as app
import openmm.unit as unit
import numpy.testing as npt
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb

from dmff import Hamiltonian


class TestADMPAPI:
    
    """ Test ADMP related generators
    """
    
    @pytest.fixture(scope='class', name='generators')
    def test_init(self):
        """load generators from XML file

        Yields:
            Tuple: (
                ADMPDispForce,
                ADMPPmeForce, # polarized
            )
        """
        rc = 4.0
        H = Hamiltonian('tests/data/admp.xml')
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
        
        yield H.getGenerators()
        
    def test_ADMPDispForce_parseXML(self, generators):
        
        gen = generators[0]
        params = gen.paramtree['ADMPDispForce']
        
        npt.assert_allclose(params['mScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(params['A'], [1203470.743, 83.2283563])
        npt.assert_allclose(params['B'], [37.81265679, 37.78544799])
        
    def test_ADMPDispForce_renderXML(self, generators):
        
        gen = generators[0]
        params = gen.paramtree['ADMPDispForce']
        gen.overwrite()
        
        assert gen.name == 'ADMPDispForce'
        npt.assert_allclose(params['mScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(params['A'], [1203470.743, 83.2283563])
        npt.assert_allclose(params['B'], [37.81265679, 37.78544799])
        
    def test_ADMPPmeForce_parseXML(self, generators):
        
        gen = generators[1]
        tree = gen.paramtree['ADMPPmeForce']
        
        npt.assert_allclose(tree['mScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(tree['pScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(tree['dScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        # Q_local is already converted to local frame
        # npt.assert_allclose(tree['Q_local'][0][:4], [-1.0614, 0.0, 0.0, -0.023671684])
        npt.assert_allclose(tree['pol'], [0.88000005, 0])
        npt.assert_allclose(tree['tholes'], [8., 0.])
        
    def test_ADMPPmeForce_renderXML(self, generators):
        
        gen = generators[1]
        params = gen.paramtree['ADMPPmeForce']
        gen.overwrite()
        
        npt.assert_allclose(params['mScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(params['pScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(params['dScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        # Q_local is already converted to local frame
        # npt.assert_allclose(params['Q_local'][0][:4], [-1.0614, 0.0, 0.0, -0.023671684])
        npt.assert_allclose(params['pol'], [0.88000005, 0])
        npt.assert_allclose(params['tholes'], [8., 0.])
        
class TestClassicalAPI:
    
    """ Test classical forcefield generators
    """
    
    @pytest.fixture(scope='class', name='generators')
    def test_init(self):
        """load generators from XML file

        Yields:
            Tuple: (
                NonBondJaxGenerator,
                HarmonicAngle,
                PeriodicTorsionForce,
            )
        """
        rc = 4.0
        H = Hamiltonian('tests/data/classical.xml')
        pdb = app.PDBFile('tests/data/linear.pdb')
        H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
        
        yield H.getGenerators()
        
    def test_Nonbond_parseXML(self, generators):
        
        gen = generators[0]
        params = gen.paramtree['NonbondedForce']
        npt.assert_allclose(params['sigma'], [1.0, 1.0, -1.0, -1.0])

        
    def test_NonBond_renderXML(self, generators):
        
        gen = generators[0]
        params = gen.paramtree['NonbondedForce']
        gen.overwrite()
        
        assert gen.name == 'NonbondedForce'
        # npt.assert_allclose(params['type'], ['n1', 'n2', 'n3', 'n4'])
        # type is asigned to the generator themself as a member variable
        npt.assert_allclose(params['charge'], [1.0, -1.0, 1.0, -1.0])
        npt.assert_allclose(params['sigma'], [1.0, 1.0, -1.0, -1.0])
        npt.assert_allclose(params['epsilon'], [0.0, 0.0, 0.0, 0.0])
        npt.assert_allclose(params['lj14scale'], [0.5])

        
    def test_HarmonicAngle_parseXML(self, generators):
        
        gen = generators[1]
        params = gen.paramtree['HarmonicAngleForce']
        npt.assert_allclose(params['k'], 836.8)
        npt.assert_allclose(params['angle'], 1.8242181341844732)

    def test_HarmonicAngle_renderXML(self, generators):
        
        gen = generators[1]
        params = gen.paramtree['HarmonicAngleForce']
        gen.overwrite()
        
        assert gen.name == 'HarmonicAngleForce'
        npt.assert_allclose(params['angle'], [1.8242181341844732] * 2)
        npt.assert_allclose(params['k'], [836.8] * 2)
        
    def test_PeriodicTorsion_parseXML(self, generators):
        
        gen = generators[2]
        params = gen.paramtree['PeriodicTorsionForce']
        npt.assert_allclose(params['prop_phase']['1'], [0])
        npt.assert_allclose(params['prop_k']['1'], [2.092])
        
    def test_PeriodicTorsion_renderXML(self, generators):
        
        gen = generators[2]
        params = gen.paramtree['PeriodicTorsionForce']
        gen.overwrite()
        assert gen.name == 'PeriodicTorsionForce'
        npt.assert_allclose(params['prop_phase']['1'], [0])
        npt.assert_allclose(params['prop_k']['1'], [2.092])
    
    def test_parse_multiple_files(self):
        pdb = app.PDBFile("tests/data/methane_water.pdb")
        h = Hamiltonian("tests/data/methane.xml", "tests/data/tip3p.xml")
        potentials = h.createPotential(pdb.topology)


def check_same_topology(top1: app.Topology, top2: app.Topology):
    assert top1.getNumChains() == top2.getNumChains(), "Number of chains are not the same"
    assert top1.getNumResidues() == top2.getNumResidues(), "Number of residues are not the same"
    assert top1.getNumAtoms() == top2.getNumAtoms(), "Number of atoms are not the same"
    assert top1.getNumBonds() == top2.getNumBonds(), "Number of bonds are not the same"
    
    atoms1, atoms2 = [at for at in top1.atoms()], [at for at in top2.atoms()]
    for i in range(top1.getNumAtoms()):
        assert atoms1[i].element == atoms2[i].element, f"Atom {i} are not the same"
    
    bonds1 = {
        (int(bond.atom1.id) - 1, int(bond.atom2.id) - 1): True \
             for bond in top1.bonds()
    }
    bonds2 = {
        (int(bond.atom1.id) - 1, int(bond.atom2.id) - 1): True \
             for bond in top2.bonds()
    }
    for key in bonds1.keys():
        if (key not in bonds2) and ((key[1], key[0]) not in bonds2):
            raise KeyError(f"Bond (key) not match")


def fix_pdb(path):
    pdb = PandasPdb()
    pdb.read_pdb(path)
    atCount = {}
    for i in range(pdb.df['HETATM'].shape[0]):
        atName = pdb.df['HETATM'].loc[i, 'atom_name']
        atCount.update({atName: atCount.get(atName, 0) + 1})
        pdb.df['HETATM'].loc[i, 'atom_name'] = f"{atName}{atCount.get(atName)}"
    pdb.to_pdb(path)


@pytest.mark.parametrize(
    "smi",
    ["CC", "CO", "C=C", "CC#N", "C1=CC=CC=C1"]
)
def test_build_top_from_rdkit(smi, tmpdir):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    key = Chem.MolToInchiKey(mol)
    wdir = Path(tmpdir).resolve()
    molfile = str(wdir / f"{key}.mol")
    pdbfile = str(wdir / f"{key}.pdb")
    Chem.MolToMolFile(mol, str(wdir / f"{key}.mol"))
    subprocess.run(["obabel", molfile, "-O", pdbfile], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fix_pdb(pdbfile)
    ref_top = app.PDBFile(pdbfile).getTopology()
    test_top = Hamiltonian.buildTopologyFromMol(mol)
    check_same_topology(ref_top, test_top)
