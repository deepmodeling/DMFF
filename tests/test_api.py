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
        params = gen.params
        
        npt.assert_allclose(params['mScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(params['A'], [1203470.743, 83.2283563])
        npt.assert_allclose(params['B'], [37.81265679, 37.78544799])
        
    def test_ADMPDispForce_renderXML(self, generators):
        
        gen = generators[0]
        xml = gen.renderXML()
        
        assert xml.name == 'ADMPDispForce'
        npt.assert_allclose(float(xml[0]['type']), 380)
        npt.assert_allclose(float(xml[0]['A']), 1203470.743)
        npt.assert_allclose(float(xml[1]['B']), 37.78544799)
        npt.assert_allclose(float(xml[1]['Q']), 0.370853)
        
    def test_ADMPPmeForce_parseXML(self, generators):
        
        gen = generators[1]
        params = gen.params
        
        npt.assert_allclose(params['mScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(params['pScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        npt.assert_allclose(params['dScales'], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        # Q_local is already converted to local frame
        # npt.assert_allclose(params['Q_local'][0][:4], [-1.0614, 0.0, 0.0, -0.023671684])
        npt.assert_allclose(params['pol'], [0.88000005, 0])
        npt.assert_allclose(params['tholes'], [8., 0.])
        
    def test_ADMPPmeForce_renderXML(self, generators):
        
        gen = generators[1]
        xml = gen.renderXML()
        
        assert xml.name == 'ADMPPmeForce'
        assert xml.attributes['lmax'] == '2'
        assert xml.attributes['mScale12'] == '0.0'
        assert xml.attributes['mScale15'] == '1.0'
        assert xml.elements[0].name == 'Atom'
        assert xml.elements[0].attributes['qXZ'] == '-0.07141020'
        assert xml.elements[2].name == 'Polarize'
        assert xml.elements[2].attributes['polarizabilityXX'][:6] == '0.8800'
        assert xml[3]['type'] == '381'
        
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
        
    def test_NonBond_parseXML(self, generators):
        
        gen = generators[0]
        params = gen.params
        npt.assert_allclose(params['sigma'], [1.0, 1.0, -1.0, -1.0])

        
    def test_NonBond_renderXML(self, generators):
        
        gen = generators[0]
        xml = gen.renderXML()
        
        assert xml.name == 'NonbondedForce'
        assert xml.attributes['lj14scale'] == '0.5'
        assert xml[0]['type'] == 'n1'
        assert xml[1]['sigma'] == '1.0'
        
    def test_HarmonicAngle_parseXML(self, generators):
        
        gen = generators[1]
        params = gen.params
        npt.assert_allclose(params['k'], 836.8)
        npt.assert_allclose(params['angle'], 1.8242181341844732)

    def test_HarmonicAngle_renderXML(self, generators):
        
        gen = generators[1]
        xml = gen.renderXML()
        
        assert xml.name == 'HarmonicAngleForce'
        assert xml[0]['type1'] == 'n1'
        assert xml[0]['type2'] == 'n2'
        assert xml[0]['type3'] == 'n3'
        assert xml[0]['angle'][:7] == '1.82421'
        assert xml[0]['k'] == '836.8'
        
    def test_PeriodicTorsion_parseXML(self, generators):
        
        gen = generators[2]
        params = gen.params
        npt.assert_allclose(params['psi1_p'], 0)
        npt.assert_allclose(params['k1_p'], 2.092)
        
    def test_PeriodicTorsion_renderXML(self, generators):
        
        gen = generators[2]
        xml = gen.renderXML()
        assert xml.name == 'PeriodicTorsionForce'
        assert xml[0].name == 'Proper'
        assert xml[0]['type1'] == 'n1'
        assert xml[1].name == 'Improper'
        assert xml[1]['type1'] == 'n1'
    
    def test_parse_multiple_files(self):
        pdb = app.PDBFile("tests/data/methane_water.pdb")
        h = Hamiltonian("tests/data/methane.xml", "tip3p.xml")
        potentials = h.createPotential(pdb.topology)
        npt.assert_allclose(
            h.getGenerators()[-1].params["charge"],
            [-0.1068, 0.0267, 0.0267, 0.0267, 0.0267, -0.834, 0.417]
        )
