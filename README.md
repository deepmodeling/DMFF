# DMFF

[![DOI: 10.1021/acs.jctc.2c01297](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.2c01297-blue)](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01297)
 
## About DMFF

**DMFF** (**D**ifferentiable **M**olecular **F**orce **F**ield) is a Jax-based python package that provides a full differentiable implementation of molecular force field models. This project aims to establish an extensible codebase to minimize the efforts in force field parameterization, and to ease the force and virial tensor evaluations for advanced complicated potentials (e.g., polarizable models with geometry-dependent atomic parameters). Currently, this project mainly focuses on the molecular systems such as: water, biological macromolecules (peptides, proteins, nucleic acids), organic polymers, and small organic molecules (organic electrolyte, drug-like molecules) etc. We support both the conventional point charge models (OPLS and AMBER like) and multipolar polarizable models (AMOEBA and MPID like). The entire project is backed by the XLA technique in JAX, thus can be "jitted" and run in GPU devices much more efficiently compared to normal python codes.

The behavior of organic molecular systems (e.g., protein folding, polymer structure, etc.) is often determined by a complex effect of many different types of interactions. The existing organic molecular force fields are mainly empirically fitted and their performance relies heavily on error cancellation. Therefore, the transferability and the prediction power of these force fields are insufficient. For new molecules, the parameter fitting process requires essential manual intervention and can be quite cumbersome. In order to automate the parametrization process and increase the robustness of the model, it is necessary to apply modern AI techniques in conventional force field development. This project serves for this purpose by utilizing the automatic differentiable programming technique to develop a codebase, which allows a more convenient incorporation of modern AI optimization techniques. It also helps the realization of many exciting functions including (but not limited to): hybrid machine learning/force field models and parameter optimization based on trajectory.

### License and credits

The project DMFF is licensed under [GNU LGPL v3.0](LICENSE). If you use this code in any future publications, please cite this using `Xinyan Wang, Jichen Li, Lan Yang, Feiyang Chen, Yingze Wang, Junhan Chang, Junmin Chen, Wei Feng, Linfeng Zhang, and Kuang Yu 
Journal of Chemical Theory and Computation 2023 19 (17), 5897-5909
DOI: 10.1021/acs.jctc.2c01297`

## User Guide

+ [1. Introduction](docs/user_guide/1.introduction.md)
+ [2. Installation](docs/user_guide/2.installation.md)
+ [3. Basic Usage](docs/user_guide/3.usage.md)
+ [4. Modules](docs/user_guide/4.modules.md)
  + [Classical](docs/user_guide/4.1classical.md)
  + [ADMP](docs/user_guide/4.2ADMPPmeForce.md)
  + [Qeq](docs/user_guide/4.3ADMPQeqForce.md)
  + [Machine Learning](docs/user_guide/4.4MLForce.md)
  + [Optimization](docs/user_guide/4.5Optimization.md)
  + [Mbar Estimator](docs/user_guide/4.6MBAR.md)
  + [OpenMM Plugin](docs/user_guide/4.7OpenMMplugin.md)
+ [5. Advanced examples](docs/user_guide/DMFF_example.ipynb)
+ [And here is a tutorial notebook of the basic usage of DMFF. Welcome to read it and get started with DMFF!](docs/user_guide/test.ipynb)

[And here is a tutorial notebook, which would tell you some basic usage of DMFF. Welcome to read it and get started witn DMFF！](docs/user_guide/test.ipynb)

## Developer Guide
+ [1. Introduction](docs/dev_guide/introduction.md)
+ [2. Software architecture](docs/dev_guide/arch.md)
+ [3. Coding conventions](docs/dev_guide/convention.md)
+ [4. Document writing](docs/dev_guide/write_docs.md)
+ [5. An example for developing: how to write a generator?](docs/dev_guide/generator.ipynb)

## Code Structure

The code is organized as follows:

+ `examples`: demos presented in Jupyter Notebook.
+ `docs`: documentation.
+ `package`: files for constructing packages or images, such as conda recipe and docker files.
+ `tests`: unit tests.
+ `dmff`: DMFF python codes
+ `dmff/api`: source code of application programming interface of DMFF.
+ `dmff/admp`: source code of automatic differentiable multipolar polarizable (ADMP) force field module.
+ `dmff/classical`: source code of classical force field module.
+ `dmff/common`: source code of common functions, such as neighbor list.
+ `dmff/sgnn`: source of subgragh neural network force field model.
+ `dmff/eann`: source of embedded atom neural network force field model.
+ `dmff/generators`: source code of force generators.
+ `dmff/operators`: source code of operators.


## Support and Contribution

Please visit our repository on [GitHub](https://github.com/deepmodeling/DMFF) for the library source code. Any issues or bugs may be reported at our issue tracker. All contributions to DMFF are welcomed via pull requests!
