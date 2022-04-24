# 1. Introduction

In this user guide, you will learn:

- Installation of DMFF
- Computation of energy and force
- Auto differentiate alone computation path
- Couple DMFF with MD engine

The first thing you should know is that DMFF is not an actual force field (such as OPLS or AMBER), but a collection of various force field (or called "potential") forms:

- Automatic Differentiable Multipolar Polarizable (ADMP)
- Classical force field derived from GAFF
- SGNN

Those modules can easily be combined with a specific research topic to create a force field. Each module has its usage and required parameters, and provide them in a specified XML file and then use them under a Python program. With DMFF, one can calculate the energy and force of a molecular mechanics system and even get the parameters' gradient. 