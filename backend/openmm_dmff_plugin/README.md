# OpenMM Plugin for DMFF


This is a plugin for [OpenMM](http://openmm.org) that used the trained JAX model by [DMFF](https://github.com/deepmodeling/DMFF) as an independent Force class for dynamics.
To use it, you need to save your DMFF model with the script in `DMFF/backend/save_dmff2tf.py`.

## Installation

### Create environment with conda
Install the python, openmm and cudatoolkit.
```shell

mkdir omm_dmff_working_dir && cd omm_dmff_working_dir
conda create -n dmff_omm -c conda-forge python=3.9 openmm cudatoolkit=11.6
conda activate dmff_omm
```
### Download `libtensorflow_cc` and install `cppflow` package
Install the precompiled libtensorflow_cc library from conda.
```shell
conda install -y libtensorflow_cc=2.9.1 -c conda-forge
```
Download the tensorflow sources file. Copy the `c` direcotry in source code to installed header files of tensorflow library, since it's needed by package `cppflow`.
```shell

wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.tar.gz
tar -xvf v2.9.1.tar.gz
cp -r tensorflow-2.9.1/tensorflow/c ${CONDA_PREFIX}/include/tensorflow/
```
Download `cppflow` and move the headers library to environment path.
```shell

git clone https://github.com/serizba/cppflow.git
cd cppflow
git apply DMFF/backend/openmm_dmff_plugin/tests/cppflow_empty_constructor.patch
mkdir ${CONDA_PREFIX}/include/cppflow
cp -r include/cppflow ${CONDA_PREFIX}/include/
```

### Install the OpenMM DMFF plugin from the source 

Compile the plugin from the source with the following steps.
1. Set up environment variables.
   ```shell
   export OPENMM_INSTALLED_DIR=$CONDA_PREFIX
   export CPPFLOW_INSTALLED_DIR=$CONDA_PREFIX
   export LIBTENSORFLOW_INSTALLED_DIR=$CONDA_PREFIX
   cd DMFF/backend/openmm_dmff_plugin/
   mkdir build && cd build
   ```

2. Run `cmake` command with the required parameters.
   ```shell
   cmake .. -DOPENMM_DIR=${OPENMM_INSTALLED_DIR} -DCPPFLOW_DIR=${CPPFLOW_INSTALLED_DIR} -DTENSORFLOW_DIR=${LIBTENSORFLOW_INSTALLED_DIR}
   make && make install
   make PythonInstall
   ```
   
3. Test the plugin in Python interface, reference platform.
   ```shell
   python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nve -n 100
   python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nvt -n 100 --platform CUDA
   ```
## Usage
Add the following lines to your Python script to use the plugin.
More details can refer to the script in `python/OpenMMDMFFPlugin/tests/test_dmff_plugin_nve.py`.

```python

from OpenMMDMFFPlugin import DMFFModel
# Set up the dmff_system with the dmff_model.    
dmff_model = DMFFModel(dp_model)
dmff_model.setUnitTransformCoefficients(1, 1, 1)
dmff_system = dmff_model.createSystem(topology)
```
