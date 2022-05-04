# 1. Introduction

In this user guide, you will learn how to:

- Install DMFF
- Compute energy and force
- Auto differentiate potential functions
- Couple DMFF with MD engine

The first thing you should know is that DMFF is not an actual force field (such as OPLS or AMBER), but a differentiable implementation of various force field (or "potential") functional forms. It contains many modules:

- ADMP module: Automatic Differentiable Multipolar Polarizable potential (MPID like potentials)
- Classical module: implements classical force fields (OPLS or GAFF like potentials)
- SGNN module: implements subgragh neural network model for intramolecular interactions

Each module implements a particular form of force field, which takes a unique set of input parameters, usually provided in a XML file. With DMFF, one can easily compute the energy as well as energy gradients including: forces, virial tensors, and gradients with respect to force field parameters etc.

