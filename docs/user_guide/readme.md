# About user guide section

In this section, you will learn:

    - Installation of DMFF
    - Computation of energy and force
    - Auto differentiate alone computation path
    - Couple DMFF with MD engine

The first thing you should know is that DMFF is not exactly a force field in the traditional sense: there is no module called *DMFF*, but a collection of potential forms. There is several forms of support in the DMFF package:

    - Automatic Differentiable Multipolar Polarizable (ADMP)
    - Classical force field derived from GAFF
    - SGNN

Those modules can easily be combined with a specific research topic to create a force field. Each module has its usage and required parameters, and provide them in a specified XML file and then use them under a Python program. With DMFF, one can calculate the energy and force of a molecular mechanics system and even get the parameters' gradient. 