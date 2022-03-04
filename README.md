# AIMD_on_QC
This is a package to perform ab initio molecular dynamics simulations on quantum computers using qiskit package.
The article can be found at https://aip.scitation.org/doi/10.1063/5.0046930
Refer to requirements.txt for the required versions of python packages.

The vv.py script contains the input for the molecule and initial conditions, performs propagation of the nuclei using velocity Verlet algorithm.
The h2_md.py script performs the quantum simulations using qiskit.
