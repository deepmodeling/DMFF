# DMFF's Manual

**DMFF** (**D**ifferentiable **M**olecular **F**orce **F**ield) is a python-based package that provides a full differentiable implementation of molecular force field calculations. This project aims to establish an extensible framework to minimize the efforts in force field parameter fitting, and to support an easy evaluation of forces and virial tensors for complicated advanced potentials (such as polarizable models with geometry-dependent atomic parameters). Currently, this project mainly focuses on the force fields of the following systems: water, biological macromolecules (peptides, proteins, nucleic acids), organic polymers, and small organic molecules (organic electrolyte, drug-like molecules) etc. And we support both the conventional point charge models (OPLS and AMBER like) and multpolar polarizable models (AMOEBA and MPID like). 

The behavior of organic molecular systems (e.g., protein folding, polymer structure, etc.) is often determined by a complex effect of many different types of interactions. The existing organic molecular force fields are mainly empirically fitted and their performance relies heavily on error cancellation. Therefore, the transferrabilities and the prediction powers of these force fields are insufficient. For new molecules, the parameter fitting process requires heavy load of manual intervention and can be quite cubersome. In order to automize the parametrization process and increase the robustness of the model, it is necessary to combine modern AI techniques with conventional force field development. This project serves for this purpose by utilizing the automatic differential programming framework to develop a toolchain, leading to many advanced functions such as: hybrid force field / machine learning models and parameter optimization based on molecular mechanics trajectory.

## User Guide

+ [1. Introduction](user_guide/introduction.md)
+ [2. Installation](user_guide/installation.md)
+ [3. Compute energy and forces](user_guide/compute.md)
+ [4. Compute gradients with auto differentiable framework](user_guide/auto_diff.md)

## Developer Guide
+ [1. Introduction](dev_guide/introduction.md)
+ [2. Architecture](dev_guide/arch.md)
+ [3. Convention](dev_guide/convention.md)

## Modules
+ [1. ADMP](modules/admp.md)


## Support and Contribution

Please visit our repository on [GitHub](https://github.com/deepmodeling/DMFF) for the library source code. Any issues or bugs may be reported at our issue tracker. All contributions to DMFF are welcomed via pull requests!
