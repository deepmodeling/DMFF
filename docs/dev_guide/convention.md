# Code Convention

In this section, you will learn:
    - How is DMFF organized
    - 

## code organization

The root directory of DMFF has following sub-directories:

    - `dmff`: source code of project
    - `docs`: documents in markdown
    - `examples`: examples can be run independently
    - `tests`: unit and integration tests

Under the `dmff`, there are several files and sub-directory:

    - `api.py`: store all the frontend modules
    - `settings.py`: global settings 
    - `utils.py`: helper functions
    — each sub-directory represents a set of potential form, e.g. `admp` is Automatic Differentiable Multipolar Polarizable, `classical` is differentiable GAFF forcefield.

