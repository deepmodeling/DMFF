# 3. Coding conventions

In this section, you will learn:

- How DMFF is organized
- Programming style recommended to follow in DMFF development

## 3.1 Code Organization

The root directory of DMFF has following sub-directories:

- `dmff`: source code of project
- `docs`: documents in markdown
- `examples`: examples can be run independently
- `tests`: unit and integration tests

Under the `dmff`, there are several files and sub-directory:

- `api.py`: API (frontend modules)
- `settings.py`: global settings 
- `utils.py`: basic functions
- each sub-directory represents a set of potential form, e.g. `admp` stands for Automatic Differentiable Multipolar Polarizable force field, and `classical` is the differentiable implementation of classical fixed-charge force field.

## 3.2 Programming Style

TBA