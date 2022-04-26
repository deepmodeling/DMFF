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
        generator = H.getGenerators()[1]
        xml = generator.renderXML()
        print(xml)