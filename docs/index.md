# Welcome to DMFF

This project aims to establish a general extensible framework to support the development of organic molecular force field and the fitting of parameters in the force field. The main target of the project include: biological macromolecules (peptides, proteins, nucleic acids, etc.), organic macromolecules, organic small molecules (including organic electrolyte, small molecule drugs), etc.

There are many factors involved in organic molecular interactions, and the behavior of organic molecular systems (such as protein folding, polymer structure, etc.) often depends on the joint influence of various interactions. The existing general organic molecular force fields (such as OPLS and amber) are mainly empirical fitting, and their portability and prediction ability are insufficient. When extended to new molecules, the parameter fitting process is cumbersome and strongly depends on error cancellation under manual intervention.

In order to accurately describe organic molecular systems, we need to accurately model various interactions within and between molecules (including long-range and short-range). Therefore, it is necessary to realize a closer combination of traditional force field and AI method, and apply AI tools to short-range potential energy surface fitting, traditional force field parameter optimization and so on. We will use the automatic differential programming framework to develop the tool chain from force field calculation to molecular mechanics simulation, so as to realize the complex functions such as traditional force field / machine learning hybrid model and parameter optimization based on molecular mechanics trajectory. Based on this project, a new generation of general organic force field database is developing, and a more automatic force field development process is established.

## Resources

[Reference Documentation](): Examples, tutorials, topic guides, and package Python APIs.
[Installation Guide](): Instructions for installing and compiling freud.

[GitHub repository](): Download the freud source code.

[Issue tracker](): Report issues or request features.

## Citation

TODO:

## Installation

### Prerequsite

DMFF depends on several packages:

* [jax]()
* [jax-md]()
* [OpenMM]()

### step-by-step

1. Install [jax](https://github.com/google/jax) (pick the correct cuda version, see more details on their installation guide):

   ```
   pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
   ```

2. Install [jax-md](https://github.com/google/jax-md) :

   ```
   pip install jax-md --upgrade
   ```

   ADMP currently relies on the space and partition modules to provide neighbor list

3. Install [OpenMM](https://openmm.org/)

    ```
    conda install -c conda-forge openmm cudatoolkit=10.0
    ```

## Example

We provide a MPID 1024 water box example. In water_1024 and water_pol_1024, we show both the nonpolarizable and the polarizable cases.

```bash
cd ./examples/water_1024
./run_admp.py

cd ./examples/water_pol_1024
./run_admp.py
```

if `DO_JIT = True`, then the first run would be a bit slow, since it tries to do the jit compilation. Further executions of `get_forces` or `get_energy` should be much faster.

## Support and Contribution

Please visit our repository on [GitHub](https://github.com/deepmodeling/DMFF) for the library source code. Any issues or bugs may be reported at our issue tracker. All contributions to DMFF are welcomed via pull requests!