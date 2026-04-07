## DMFF Codebase High-Level Architecture Overview

This document targets humans/agents who need to develop, refactor, test, or debug in the DMFF codebase. It summarizes the project purpose, core architecture, build and test flows, and configuration/security information that is explicitly present in the repository.

> All descriptions here are derived from this repository itself (for example `README.md`, `docs/`, `dmff/`, `backend/`), without introducing external or generic development practices.

---

### 1. Project Overview

- **Goal and Scope**  
  - DMFF (Differentiable Molecular Force Field) is a JAX-based Python package that provides a fully differentiable implementation of molecular force-field models, enabling parameter optimization and efficient energy/force evaluation for systems such as water, biomacromolecules, organic polymers, and small organic molecules `README.md:5`, `docs/index.md:5`.
  - It supports conventional point-charge models (OPLS/AMBER-like) and multipolar polarizable models (AMOEBA/MPID-like), and is designed to integrate modern machine-learning optimization techniques for automated parameterization and trajectory-based optimization `README.md:7-10`, `docs/dev_guide/introduction.md:11-19`.

- **Core Layered Architecture (Python)**
  - **Top-level package entry `dmff`**  
    - `dmff/__init__.py:1-6` re-exports key objects:
      - Global settings: `dmff.settings` (numeric precision, JIT flag, debug flag);
      - Neighbor-list utilities: `dmff.common.nblist.NeighborList` / `NeighborListFreud`;
      - Force-field generators: `dmff.generators`;
      - System Hamiltonian: `dmff.api.Hamiltonian`;
      - Topology operators and MD tools: `dmff.operators`, `dmff.mdtools`.
  - **API layer: Hamiltonian & topology**  
    - `dmff/api/__init__.py:1-2` exposes two core classes: `Hamiltonian` and `DMFFTopology`.
      - `Hamiltonian` encapsulates total energy/force, etc.;
      - `DMFFTopology` represents topology and parameter data for a system.
  - **Generators and Calculators**  
    - `dmff/generators/__init__.py:1-4` aggregates the submodules `classical`, `admp`, `ml`, and `qeq`, each corresponding to a particular potential form.
    - `docs/dev_guide/introduction.md:11-18` describes the division of responsibilities:
      - `Generator` loads and organizes parameters from force-field XML files;
      - **Calculators** are pure, heavy-duty functions that take atomic positions and force-field parameters as input and return energies; they are JAX-differentiable and JIT-compilable.
  - **Runtime settings**  
    - `dmff/settings.py:3-7` defines:
      - `PRECISION` (e.g. `'double'`) controlling JAX 64-bit precision;
      - `DO_JIT` controlling whether to JIT-compile core computations;
      - `DEBUG` controlling debug behavior;
      - `update_jax_precision()` updates the global JAX `jax_enable_x64` flag at import time `dmff/settings.py:10-19`.
  - **Operators pipeline**  
    - `dmff/operators/base.py:4-12` defines `BaseOperator`:
      - `__call__` accepts a `DMFFTopology` and delegates to `operate`;
      - subclasses in `dmff/operators/` implement topology/parameter transformations (e.g. typing, virtual sites, AM1 charges) in a pipeline-like fashion.
  - **Neighbor list and backend acceleration**  
    - Python-level neighbor list: `dmff/common/nblist.py` (not expanded here, but exported in `dmff/__init__.py:2`).
    - C++/CUDA backend: `dmff/dpnblist/` provides a high-performance neighbor-list library with CPU/GPU scheduling algorithms and doctest-based tests `dmff/dpnblist/tests/CMakeLists.txt:1-38`.

- **ADMP and Classical force-field architecture (from docs)**  
  - `docs/assets/DMFF_arch.md:1-26` outlines a three-part architecture:
    - **Parser & typification**:
      - Input: force-field XML file;
      - `parseElement` parses XML and builds **Generators**;
      - `createPotential` produces an intermediate representation containing atomic/topological parameters.
    - **Calculators layer**:
      - ADMP: General Pairwise Calculator, Multipole PME Calculator, Dispersion PME Calculator;
      - Classical: Intramolecular and Intermolecular calculators;
      - All calculators expose a unified energy API `potential(pos, box, pairs, params)` which is JAX-differentiable.
    - **Neighbor list and parameter coupling**:
      - Neighbor pairs `pairs` come from a jax-md-style neighbor list (edge `J -> I` in the diagram);
      - Generator outputs (differentiable parameters) feed into the calculators.

- **OpenMM plugin backend**  
  - `backend/openmm_dmff_plugin/README.md:1-4` describes an OpenMM plugin that embeds a trained DMFF JAX model as an OpenMM `Force` for molecular dynamics.
  - Installation requires `libtensorflow_cc`, `cppflow`, and CMake; the plugin builds `DMFFForce`-related kernels under `backend/openmm_dmff_plugin/openmmapi/` and `backend/openmm_dmff_plugin/platforms/` (file-level details are omitted here).

- **Documentation and examples**  
  - Documentation: MkDocs-based site with navigation defined in `mkdocs.yml:1-27` and `docs/index.md:17-39`.
  - Examples: `examples/` provides runnable examples for Classical, ADMP, MLForce, OpenMM plugin, DiffTraj, etc. `README.md:48-57`, `docs/user_guide/3.usage.md`.

Currently there is no existing `AGENT.md` / `AGENTS.md`, and no `.cursor/rules/`, `.trae/rules/`, or `.github/copilot-instructions.md` files have been detected in this repository.

---

### 2. Build & Commands

This section lists only commands and tools that appear explicitly in the repository.

- **Install from source (Python package)**  
  - Installing DMFF from source `docs/user_guide/2.installation.md:35-40`:
    - `git clone https://github.com/deepmodeling/DMFF.git`
    - `cd DMFF`
    - `pip install . --user`
  - Dependency installation (partial):
    - Conda environment creation and installation of JAX, mdtraj, optax, jaxopt, pymbar, OpenMM, RDKit, etc. See `docs/user_guide/2.installation.md:3-33` for exact commands.

- **Python tests (pytest)**  
  - The root `Makefile:1-31` defines per-module pytest targets, all using `pytest --disable-warnings`:
    - `make test_admp` → `pytest --disable-warnings tests/test_admp`;
    - `make test_classical` → `pytest --disable-warnings tests/test_classical`;
    - `make test_common` → `pytest --disable-warnings tests/test_common`;
    - `make test_difftraj` → `pytest --disable-warnings tests/test_difftraj`;
    - `make test_dimer` → `pytest --disable-warnings tests/test_dimer`;
    - `make test_frontend` → `pytest --disable-warnings tests/test_frontend`;
    - `make test_mbar` → `pytest --disable-warnings tests/test_mbar`;
    - `make test_sgnn` → `pytest --disable-warnings tests/test_sgnn`;
    - `make test_energy` → `pytest --disable-warnings tests/test_energy.py`;
    - `make test_utils` → `pytest --disable-warnings tests/test_utils.py`.

- **C++ neighbor-list backend (dpnblist) build & tests**  
  - `dmff/dpnblist/CMakeLists.txt` (not expanded here) defines how to build the C++/CUDA neighbor-list library.
  - Test executable `dpnblist_test` is defined in `dmff/dpnblist/tests/CMakeLists.txt:1-38`:
    - `add_executable(dpnblist_test ...)` with multiple `test_*.cpp` sources;
    - `find_package(doctest)` or a bundled `doctest.cmake` is used to enable doctest-based unit tests.

- **OpenMM DMFF plugin build & tests**  
  - Environment setup and build commands are documented in `backend/openmm_dmff_plugin/README.md:9-56`:
    - Install `python`, `openmm`, `cudatoolkit`, and `libtensorflow_cc` via conda;
    - Download TensorFlow sources and copy `tensorflow/c` headers into the conda environment to satisfy `cppflow` requirements;
    - Set `OPENMM_INSTALLED_DIR`, `CPPFLOW_INSTALLED_DIR`, `LIBTENSORFLOW_INSTALLED_DIR`;
    - In `backend/openmm_dmff_plugin/build`, run `cmake .. -DOPENMM_DIR=... -DCPPFLOW_DIR=... -DTENSORFLOW_DIR=...` and then `make && make install && make PythonInstall`.
  - Python-level plugin tests `backend/openmm_dmff_plugin/README.md:58-62`:
    - `python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nve -n 100`
    - `python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nvt -n 100 --platform CUDA`

- **Docs development & preview (MkDocs)**  
  - Documentation framework: MkDocs `docs/dev_guide/write_docs.md:5`.
  - Preview command `docs/dev_guide/write_docs.md:27-31`:
    - In the directory containing `mkdocs.yml`, run `mkdocs serve` to start a local dev server with auto-reload.

---

### 3. Code Style

This section collects only style guidelines that are explicitly stated in the repository.

- **Code organization**  
  - Root layout `docs/dev_guide/convention.md:10-15`:
    - `dmff/`: project source code;
    - `docs/`: Markdown documentation;
    - `examples/`: standalone examples;
    - `tests/`: unit and integration tests.
  - Within `dmff/` `docs/dev_guide/convention.md:17-22`:
    - `api.py`: API (frontend modules);
    - `settings.py`: global settings;
    - `utils.py`: basic utilities;
    - each subdirectory corresponds to a potential form (e.g. `admp`, `classical`).

- **Docstrings and comments**  
  - DMFF adopts **NumPy-style docstrings**:
    - `docs/dev_guide/convention.md:24-27` states:
      - methods and classes should use NumPy-style docstrings, combined with `typing` annotations, to support API documentation generation;
      - an extended example is provided via the Napoleon NumPy-style sample `docs/dev_guide/convention.md:30-387`.
  - Documentation system: MkDocs; authoring guidelines in `docs/dev_guide/write_docs.md`:
    - new docs are added as Markdown files in the appropriate directories;
    - images should be placed under `docs/assets/` and referenced via relative paths `docs/dev_guide/write_docs.md:21-23`.

- **Language and dependencies**  
  - Python version and runtime dependencies:
    - `setup.py:47-53` requires Python `~=3.8`, and `setup.py:21-30` lists core dependencies such as `numpy>=1.18`, `jax>=0.4.1`, `openmm>=7.6.0`, `freud-analysis`, `networkx>=3.0`, `optax>=0.1.4`, `jaxopt>=0.8.0`, `pymbar>=4.0.0`, and `tqdm`.
  - Docs-related dependencies: `requirements.txt:1-15` lists documentation tooling (`mkdocs`, `mkdocs-autorefs`, `mkdocs-gen-files`, `mkdocs-literate-nav`, `mkdocstrings`, `mkdocstrings-python`, `pygments`) and runtime libraries (`jax`, `jaxlib`, `pymbar`, `rdkit`, `ase`).

---

### 4. Testing

DMFF uses both Python-level unit/integration tests and C++-level backend tests.

- **Python test layout**  
  - All Python tests live under `tests/`, covering: frontend API, classical force fields (`tests/test_classical/`), ADMP module (`tests/test_admp/`), neighbor-list utilities (`tests/test_common/`), DiffTraj, MBAR, SGNN, EANN, and others (see the directory tree under `tests/`).
  - The `Makefile:1-31` provides per-module pytest entry points, making it easy to run only a subset of tests.
  - `tests/conftest.py:1` is currently empty; there are no repository-wide pytest fixtures or hooks defined.

- **Installation sanity checks (Python)**  
  - User guide installation check `docs/user_guide/2.installation.md:42-52`:
    - In an interactive Python session, import `dmff` and `dmff.admp` to ensure the package is available;
    - run `examples/water_fullpol/run.py` to confirm example scripts execute successfully.

- **C++ backend tests (dpnblist)**  
  - `dmff/dpnblist/tests/CMakeLists.txt:1-38`:
    - defines the `dpnblist_test` executable combining multiple `test_*.cpp` files;
    - uses doctest for unit tests; when an external doctest installation is not found, it pulls in `external/doctest-2.4.11` and uses `doctest.cmake`;
    - `doctest_discover_tests` registers tests with CTest.

- **OpenMM plugin tests**  
  - Python-level tests `backend/openmm_dmff_plugin/README.md:58-62`:
    - `python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nve -n 100`;
    - `python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nvt -n 100 --platform CUDA`.
  - C++-level tests include `TestDMFFPlugin4CUDA.cpp` and `TestDMFFPlugin4Reference.cpp` with corresponding `CMakeLists.txt` under `backend/openmm_dmff_plugin/platforms/*/tests/`, used to validate force and energy consistency across platforms.

---

### 5. Security

The repository does not contain a dedicated security design document or explicit security policies. This section lists only direct, observable facts related to data and dependencies, without adding generic recommendations.

- **Data and file types**  
  - Force fields and topologies: many XML and PDB files under `tests/data/` and `examples/` provide input for topology and parameter construction (`tests/data/*.xml`, `examples/*/*.xml`, `*.pdb`, etc.).
  - Trained models and parameters:
    - ML force-field parameters are stored in files such as `*.pickle` and `*.pt`, e.g. `examples/eann/eann_model.pickle`, `examples/sgnn/test_backend/model1.pth`, `tests/data/water_eann.pickle`;
    - the OpenMM plugin uses `backend/save_dmff2tf.py` to export JAX models into a TensorFlow-compatible format consumed by the plugin `backend/openmm_dmff_plugin/README.md:4-5`.

- **Dependencies and runtime environment**  
  - Core numerical dependencies include `jax`, `jaxlib`, `numpy`, `openmm`, `rdkit`, etc., with version constraints specified in `setup.py:21-30`, `requirements.txt:1-15`, and `docs/user_guide/2.installation.md:3-33`.
  - The OpenMM plugin depends on `libtensorflow_cc` and `cppflow`; TensorFlow headers under `tensorflow/c` must be copied from upstream sources into the conda environment `backend/openmm_dmff_plugin/README.md:18-27`.

- **Access control and encryption**  
  - No additional access-control, authentication, or encryption mechanisms are defined in this repository;
  - Configuration example `config/freud.ini:2-41` is for a third-party tool (layout, key bindings, DB filename, color scheme) and is not coupled to DMFF’s internal numerical logic.

More fine-grained security policies (data isolation, permission control, etc.) need to be handled by the systems that integrate DMFF; this repository itself does not impose additional constraints.

---

### 6. Configuration & Environment

- **Python environment and dependencies**  
  - Recommended setup in `docs/user_guide/2.installation.md:3-33`:
    - create a conda environment named `dmff` (example uses Python 3.9);
    - install specific versions of JAX (CPU or CUDA builds), mdtraj, optax, jaxopt, pymbar, OpenMM, RDKit, etc.
  - Package-level dependencies are centralized in `setup.py:21-30` and `requirements.txt:1-15` to aid environment reproduction.

- **Global numerical and debug settings**  
  - `dmff/settings.py:3-19` defines runtime configuration:
    - `PRECISION`: controls whether JAX double precision is enabled (via `update_jax_precision` updating `jax_enable_x64`);
    - `DO_JIT`: controls JIT compilation of core computations;
    - `DEBUG`: toggles debug behavior;
    - these are exported via `dmff/__init__.py:1` and can be imported and modified by user code.

- **Documentation system configuration**  
  - `mkdocs.yml:1-47`:
    - defines the site name (`DMFF`) and navigation (User Guide, Developer Guide, module docs, etc.);
    - uses the `readthedocs` theme and `pymdownx.arithmatex` for math rendering;
    - enables `gen-files`, `literate-nav`, and `mkdocstrings` plugins to generate API references and SUMMARY-based navigation.

- **Docker environments (packaging & development)**  
  - `package/docker/develop_cpu.dockerfile` and `package/docker/develop_gpu.dockerfile` describe CPU/GPU development images (system dependencies plus Python environment) for reproducible development inside containers.

- **External tool configuration example**  
  - `config/freud.ini:2-41` configures a third-party tool (likely an HTTP/requests inspector) with layout, key bindings, database filename, and style settings; it is independent of DMFF’s core simulation logic and can be treated as optional tooling.

---

The above content is intended to let developers or agents grasp DMFF’s overall design, build process, and testing/configuration strategy without fully reading the code. For concrete implementation or refactoring tasks, combine this overview with the corresponding module-specific docs (for example `docs/dev_guide/` and `docs/user_guide/4.*.md`) and nearby tests to drive detailed understanding and validation.

