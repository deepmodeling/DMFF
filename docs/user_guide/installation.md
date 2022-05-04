# 2. Installation
## 2.1 Install Dependencies
+ Install [jax](https://github.com/google/jax) (select the correct cuda version, see more details in the Jax installation guide):
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
## 2.2 Install DMFF from Source Code
One can download the DMFF source code from github:
```bash
git clone https://github.com/deepmodeling/DMFF.git
```
Then you may install DMFF using `pip`:
```bash
cd dmff
pip install . --user
```

## 2.3 Test Installation
To test if DMFF is correctly installed, you can run the following commands in an interactive python session:
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
Note that the scripts will run slower than expect if `DO_JIT = True` in `dmff/settings.py`. This is because the programm will do the jit compilation when a function is invoked in the first time.
