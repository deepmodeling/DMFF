import numpy as np
from scipy import constants


SQRT_PI = np.sqrt(np.pi)

J2EV = constants.physical_constants["joule-electron volt relationship"][0]
# from kJ/mol to eV/particle
ENERGY_COEFF = J2EV * constants.kilo / constants.Avogadro

# vacuum electric permittivity in eV^-1 * angstrom^-1
EPSILON = constants.epsilon_0 / constants.elementary_charge * constants.angstrom
# DIELECTRIC = 1389.35455846
DIELECTRIC = 1 / (4 * np.pi * EPSILON) / ENERGY_COEFF
