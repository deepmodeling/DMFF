import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
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

