# DMFF's documentation

**DMFF** (**D**ifferentiable **M**olecular **F**orce **F**ield) is a package written in Python, designed to implement differentiable molecular force field calculations. This project aims to establish a general extensible framework to support the development of force fields by minimizing the effort required during the parameters fitting process. Currently, the project mainly focuses on development of force fields that describe the following systems: water, biological macromolecules (peptides, proteins, nucleic acids), organic polymers (PEG) and small organic molecules (organic electrolyte, drug-like molecules).

There are many factors involved in organic molecular interactions, and the behavior of organic molecular systems (such as protein folding, polymer structure, etc.) often depends on the joint influence of various interactions. The existing general organic molecular force fields (such as OPLS and amber) are mainly empirical fitting, and their portability and prediction ability are insufficient. When extended to new molecules, the parameter fitting process is cumbersome and strongly depends on error cancellation under manual intervention.

In order to accurately describe organic molecular systems, we need to accurately model various interactions within and between molecules (including long-range and short-range). Therefore, it is necessary to realize a closer combination of traditional force field and AI method, and apply AI tools to short-range potential energy surface fitting, traditional force field parameter optimization and so on. We will use the automatic differential programming framework to develop the tool chain from force field calculation to molecular mechanics simulation, so as to realize the complex functions such as traditional force field / machine learning hybrid model and parameter optimization based on molecular mechanics trajectory. Based on this project, a new generation of general organic force field database is developing, and a more automatic force field development process is established.

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