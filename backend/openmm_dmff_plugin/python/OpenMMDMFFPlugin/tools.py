from __future__ import absolute_import
try:
    from openmm import app, KcalPerKJ
    import openmm as mm
    from openmm import unit as u
    from openmm.app import *
    import openmm.unit as unit
except:
    from simtk import unit as u
    import simtk.openmm as mm
    from simtk.openmm.app import *
    import simtk.openmm as mm
    import simtk.unit as unit
    
import sys
from datetime import datetime, timedelta
import numpy as np

try:
    string_types = (unicode, str)
except NameError:
    string_types = (str,)

from .OpenMMDMFFPlugin import DMFFForce

class ForceReporter(object):
    def __init__(self, file, group_num, reportInterval):
        self.group_num = group_num
        if self.group_num is None:
            self._out = open(file, 'w')
            #self._out.write("Get the forces of all components"+"\n")
        else:
            self._out = open(file, 'w')
            #self._out.write("Get the forces of group "+str(self.group_num) + "\n") 
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        # return (steps, positions, velocities, forces, energies)
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        if self.group_num is not None:
            state = simulation.context.getState(getForces=True, groups={self.group_num})
        else:
            state = simulation.context.getState(getForces=True)
        forces = state.getForces().value_in_unit(u.kilojoules_per_mole/u.nanometers)
        self._out.write(str(forces)+"\n")



class DMFFModel():
    def __init__(self, model_file) -> None:
        self.model_file = model_file
        self.dmff_force = DMFFForce(model_file)
        return
    
    def setUnitTransformCoefficients(self, coordinatesCoefficient, forceCoefficient, energyCoefficient):
        """Set the unit transform coefficients for the DMFF model.
        Within the OpenMM context, the units for coordinates/forces/energy are restricted to nm and kJ/(mol * nm) and kJ/mol, respectively.

        Args:
            coordinatesCoefficient (float): Coefficient for input coordinates that transforms the units of the coordinates from nanometers to the units required by the DMFF model.
            forceCoefficient (float): Coefficient for forces that transforms the units of the DMFF calculated forces to the units used by OpenMM (kJ/(mol * nm)).
            energyCoefficient (float): Coefficient for energy that transforms the units of the DMFF calculated energy to the units used by OpenMM (kJ/mol).
        """
        self.dmff_force.setUnitTransformCoefficients(coordinatesCoefficient, forceCoefficient, energyCoefficient)
        return
    
    def setHasAux(self, has_aux = False):
        """Set whether the DMFF model has auxilary output.
        Used when model was saved with has_aux = True.

        Args:
            has_aux (bool, optional): Defaults to False.
        """
        self.dmff_force.setHasAux(has_aux)
        return
    
    def setCutoff(self, cutoff = 1.2):
        """Set the cutoff for the DMFF model.

        Args:
            cutoff (float, optional): Defaults to 1.2.
        """
        self.dmff_force.setCutoff(cutoff)
        return
    
    def createSystem(self, topology):
        """Create the OpenMM System object for the DMFF model.

        Args:
            topology (_type_): OpenMM Topology object
        
        """
        dmff_system = mm.System()
        
        # Add particles into force.
        for atom in topology.atoms():
            dmff_system.addParticle(atom.element.mass)
    
        dmff_system.addForce(self.dmff_force)
        
        return dmff_system