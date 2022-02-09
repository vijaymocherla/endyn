# PyCI

`PyCI` is a python codebase that demonstrates different configuration interaction calculations. The aim here is to implement and document differnt CI methods as I learn them. References to the source code and research papers are provided where needed.  

## Notes
- Calculation of One and Two electron integrals in AO basis.: Currently uses MintsHelper from `psi4 --version 1.5`
- Generation of CI matrix: Currently formula's from `Szabo and Ostulund 1996` are used adopted for CI-S,D
- An implementation of CIS(D) for excited-state calculations   

    - TODO: 
        - Add in-built scf procedures with pulay's DIIS
        - A method that demonstrates how to compute CI matrix elements with a bit string implementation of Slater-Condon rules.  
        - Incorporate a submodule to compute Spin-adapted configuration state functions(CSFs) for S and D (T, Q etc for diatomic systems) using spin-eigen functions for CSF states. 
        - A submodule for spin-blocked CIS matrices and other variations.
        - Add a way to incorporate PySCF ( introduce an option to use `libcint` with `pyscf` will be added)

    - HELPER :
        - I/O module to handle input/output with the package
            - Read molecular geometry into the program from .xyz files
            - Generate output at different steps of the calculations for various modules.
        - Cubeprop module to generate .cube files with densites.
        - Eigensolver method, to get CI eigenvalues and eigenvectors (also to get ground state for time propagations)
        - Calculate dipole integrals interfaced with psi4 or pyscf
        