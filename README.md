# DMFF

**DMFF** (**D**ifferentiable **M**olecular **F**orce **F**ield) is a Jax-based python package that provides a full differentiable implementation of molecular force field models. This project aims to establish an extensible codebase to minimize the efforts in force field parameterization, and to ease the force and virial tensor evaluations for advanced complicated potentials (e.g., polarizable models with geometry-dependent atomic parameters). Currently, this project mainly focuses on the molecular systems such as: water, biological macromolecules (peptides, proteins, nucleic acids), organic polymers, and small organic molecules (organic electrolyte, drug-like molecules) etc. We support both the conventional point charge models (OPLS and AMBER like) and multipolar polarizable models (AMOEBA and MPID like). The entire project is backed by the XLA technique in JAX, thus can be "jitted" and run in GPU devices much more efficiently compared to normal python codes.

The behavior of organic molecular systems (e.g., protein folding, polymer structure, etc.) is often determined by a complex effect of many different types of interactions. The existing organic molecular force fields are mainly empirically fitted and their performance relies heavily on error cancellation. Therefore, the transferability and the prediction power of these force fields are insufficient. For new molecules, the parameter fitting process requires essential manual intervention and can be quite cumbersome. In order to automate the parametrization process and increase the robustness of the model, it is necessary to apply modern AI techniques in conventional force field development. This project serves for this purpose by utilizing the automatic differentiable programming technique to develop a codebase, which allows a more convenient incorporation of modern AI optimization techniques. It also helps the realization of many exciting functions including (but not limited to): hybrid machine learning/force field models and parameter optimization based on trajectory.

## User Guide

+ [1. Introduction](user_guide/introduction.md)
+ [2. Installation](user_guide/installation.md)
+ [3. Compute energy and forces](user_guide/compute.md)
+ [4. Compute gradients with auto differentiable framework](user_guide/auto_diff.md)
+ [5. Theories](user_guide/theory.md)
+ [6. Introduction to force field xml files](user_guide/xml_spec.md)

## Developer Guide
+ [1. Introduction](dev_guide/introduction.md)
+ [2. Architecture](dev_guide/arch.md)
+ [3. Convention](dev_guide/convention.md)

## Modules
+ [1. ADMP](modules/admp.md)


## Support and Contribution

Please visit our repository on [GitHub](https://github.com/deepmodeling/DMFF) for the library source code. Any issues or bugs may be reported at our issue tracker. All contributions to DMFF are welcomed via pull requests!
