# ENDyn

`ENDyn` is a python package for simulating electron-nuclear dynamics. 

References to the source code and research papers are provided where needed.  

## Installation 
To install `ENDyn`, clone this repository and compile it using the `setup.py` file. Make sure to get the dependencies mentioned here. It is recommended that you use the conda recipe provided here.

This package is still under development, so please report any bugs by opening an issue in this repo. 

Note: The following installation guide assumes that you are running a *NIX OS (MacOS or Linux). 

### Dependencies
- It is recommended that you use intel compilers and the math kernel library (MKL) if you're running this on intel hardware. (See [intel one api toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) ) 
- CMake (version 3.15 or higher) 
- Psi4 (an updated version can be found here [psi4/psi4](https://github.com/psi4/psi4))
- pybind11
- eigen3 
- NumPy 
- scipy
- opt_einsum
- threadpoolctl

### Conda Recipe
If you plan to use the conda package manager all the above dependencies can be installed using the `environment.yml`. 
- Install the [Conda package manager](https://docs.conda.io/en/latest/miniconda.html)
- Setup `ENDyn` using the following recipe.
```sh
cd endyn
conda env create -f environment.yml
conda activate endyn
pip install -e .
```
 
