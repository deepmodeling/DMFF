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
    
    def test_admp_pme(self):
        
        rc = 4.0
        H = Hamiltonian("tests/data/admp.xml")
        pdb = app.PDBFile('tests/data/waterbox_31ang.pdb')
        system = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
        generators = H.getGenerators()
        
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
 
    def test_nonbond(self):
        
        H = Hamiltonian("tests/data/coul2.xml")
        pdb = app.PDBFile('tests/data/lj2.pdb')
        system = H.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        generators = H.getGenerators()
        xml = generators[0].renderXML()
        assert xml.name == 'NonbondedForce'
        assert xml.attributes['lj14scale'] == '0.5'
        assert xml[0]['type'] == 'n1'
        assert xml[1]['sigma'] == '1.0'

    def test_HarmonicAngleJaxGenerator(self):
        
        H = Hamiltonian('tests/data/angle1.xml')
        pdb = app.PDBFile('tests/data/angle1.pdb')
        system = H.createPotential(pdb.topology)
        generators = H.getGenerators()
        