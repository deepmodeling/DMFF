# 2. Installation
## 2.1 Install dependencies
+ Install [jax](https://github.com/google/jax) (pick the correct cuda version, see more details on their installation guide):
```bash
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
+ Install [jax-md](https://github.com/google/jax-md):
```bash
pip install jax-md
```
+ Install [OpenMM](https://openmm.org/):
```bash
conda install -c conda-forge openmm
```
## 2.2 Install DMFF from source code
One can download the source code of DMFF by
```bash
git clone https://github.com/deepmodeling/DMFF.git
```
then you may install DMFF easily by:
```bash
cd dmff
pip install . --user
```

## 2.3 Test installation
To test if the DMFF is correctly installed, you can run the following commands in a Python interpreter:
```python
>>> import dmff
>>> import dmff.admp
```

You can also run the example scripts to test whether DMFF is installed correctly.
```bash
cd ./examples/water_1024
python ./run_admp.py

cd ./examples/water_pol_1024
python ./run_admp.py
```
Note that the first run of the scripts will be a little bit slow if `DO_JIT = True` in `dmff/settings.py`. This is because the programm will try to do the jit compilation.