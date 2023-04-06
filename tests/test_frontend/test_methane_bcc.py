import openmm.app as app
import openmm.unit as unit
from dmff.api.topology import DMFFTopology
from dmff.api.paramset import ParamSet
from dmff.operators import TemplateVSiteOperator, SMARTSVSiteOperator, TemplateATypeOperator, SMARTSATypeOperator, AM1ChargeOperator, GAFFTypeOperator
from dmff.api.xmlio import XMLIO
from dmff.generators.classical import CoulombGenerator, LennardJonesGenerator
import numpy as np
import jax.numpy as jnp
from dmff.settings import update_jax_precision
update_jax_precision("double")


def test_methane():
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/ethane_smarts_newapi.xml")
    ffinfo = xmlio.parseXML()
    print(ffinfo["Forces"]["CoulombForce"])


if __name__ == "__main__":
    test_methane()