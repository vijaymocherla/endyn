# PyTDCI

`PyTDCI` is a python package to perform *time-dependent configuration interaction*(TDCI) calculations.

## Modules
CORE :
- Calculation of One and Two electron integrals in AO basis.
    - Psi4 (The initial build will be based on `psi4 --version 1.5`)
    - PySCF (Eventually an option to use `libcint` with `pyscf` will be added)
- SCF procedure to calculate Hartree-Fock ground state(HF gs)
    - The initial build(v1.0) will use currently available procedure in `psi4` with scripts from helper functions module  
    - Future builds :
        - Add in-built scf procedures with pulay's DIIS
- Generation of CI matrix
    - In v1.0 direct formula's from `Szabo and Ostulund 1996` will be adopted for CI-S,D
    - Future builds: 
        - In v1.2 a method to compute CI matrix elements with a bit string implementation of Slater-Condon rules.  
        -  Plans to incorporate a submodule to compute Spin-adapted configuration state functions(CSFs) for S and D (T, Q etc for diatomic systems) using spin-eigen functions for CSF states. 
        - A submodule for spin-blocked CIS matrices and other variations.
- Time Propagation 
    - RK4 (a simple rk4 implementation)
    - Future builds:
        - A spectral propagation method to benchmark numerical solvers.   
        -  In v1.2 can be added to make larger propagations times more efficent with variable time-stepping procedures such `DOPRI8` (Dormand-Prince method)


HELPER :
- I/O module to handle input/output with the package
    - Read molecular geometry into the program from .xyz files
    - Generate output at different steps of the calculations for various modules.
- Charge module to get Mulliken & Lowdin charges and other properties. 
    - Interface with time propagations algorithms.
- Cubeprop module to generate .cube files with densites.
- Eigensolver method, to get CI eigenvalues and eigenvectors (also to get ground state for time propagations)
- Fields module to add interactions to the molecular hamiltonian.
    - Calculate dipole integrals interfaced with psi4 or pyscf
    - Static fields, Cos^2 pulse, Chirped pulses.