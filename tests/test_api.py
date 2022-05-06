import jax.numpy as jnp
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import numpy.testing as npt
from dmff.api import Hamiltonian
import dmff.api as api
import pytest

class TestApiXMLRender:
    
    @pytest.fixture(scope='class', name='generators')
    def test_init(self):
        rc = 4.0
        H = Hamiltonian("tests/data/admp.xml")
        pdb = app.PDBFile('tests/data/waterbox_31ang.pdb')
        system = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
        generators = H.getGenerators()
        yield generators
    
    def test_admp_pme(self, generators):
        xml = generators[1].renderXML()
        assert xml.name == 'ADMPPmeForce'
        assert xml.attributes['lmax'] == '2'
        assert xml.attributes['mScale12'] == '0.0'
        assert xml.attributes['mScale15'] == '1.0'
        
        assert xml.elements[0].name == 'Atom'
        assert xml.elements[0].attributes['qXZ'] == '-0.07141020'
        
        assert xml.elements[2].name == 'Polarize'
        assert xml.elements[2].attributes['polarizabilityXX'] == '0.88000000'
        
        assert xml[3]['type'] == '381'
 
    def test_admp_disp(self):
        pass