# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem

from qiskit import Aer
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.algorithms.optimizers import L_BFGS_B, SLSQP
from qiskit_nature.circuit.library import HartreeFock, UCC
from qiskit.algorithms import VQE
from qiskit.algorithms import NumPyMinimumEigensolver


def compute_energy_and_grad(r, dx, \
        shots,  qubit_reduction=False, backend_type='statevector_simulator',
        map_type="JORDAN_WIGNER", parameters=None, opt=SLSQP()):

    def get_h2_molecule(r):
        basis = 'sto-3g'
        atom='H 0 0 0; H 0 0 {}'.format(r)
        driver = PySCFDriver(atom=atom,
                        unit=UnitsType.BOHR,
                        basis=basis)
        problem = ElectronicStructureProblem(driver)

        molecule = driver.run()
        return molecule, problem
        
    print("r:", r)

    # r
    molecule, problem = get_h2_molecule(r)
    e_nr = molecule.nuclear_repulsion_energy
    second_q_ops = problem.second_q_ops()
    main_op = second_q_ops[0]

    # r - dx
    rmdx = r - dx
    molecule_mdx, problem_mdx = get_h2_molecule(rmdx)
    e_nr_mdx = molecule_mdx.nuclear_repulsion_energy
    second_q_ops_mdx = problem_mdx.second_q_ops()
    main_op_mdx = second_q_ops_mdx[0]

    # r + dx
    rpdx = r + dx
    molecule_pdx, problem_pdx = get_h2_molecule(rpdx)
    e_nr_pdx = molecule_pdx.nuclear_repulsion_energy
    second_q_ops_pdx = problem_pdx.second_q_ops()
    main_op_pdx = second_q_ops_pdx[0]
    
    num_particles = (problem.molecule_data_transformed.num_alpha,
                    problem.molecule_data_transformed.num_beta)
    num_spin_orbitals = 2 * problem.molecule_data.num_molecular_orbitals

    mapper = JordanWignerMapper()
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)

    # map to qubit operators
    qubit_op = converter.convert(main_op, num_particles=num_particles)
    qubit_op_mdx = converter.convert(main_op_mdx, num_particles=num_particles)
    qubit_op_pdx = converter.convert(main_op_pdx, num_particles=num_particles)

    # Exact solution
    numpy_solver = NumPyMinimumEigensolver()
    ret_exact = numpy_solver.compute_minimum_eigenvalue(qubit_op)
    ret_exact_mdx = numpy_solver.compute_minimum_eigenvalue(qubit_op_mdx)
    ret_exact_pdx = numpy_solver.compute_minimum_eigenvalue(qubit_op_pdx)
    e_fci = ret_exact._eigenvalue.real + e_nr
    e_fci_mdx = ret_exact_mdx.eigenvalue.real + e_nr_mdx
    e_fci_pdx = ret_exact_pdx.eigenvalue.real + e_nr_pdx
    print("Exact energy:", e_fci)
    grad_fci = (e_fci_pdx - e_fci_mdx) / 2 / dx

    # setup the initial state for the ansatz
    init_state = HartreeFock(num_spin_orbitals, num_particles, converter)

    backend = Aer.get_backend('statevector_simulator')
    ansatz = UCC(num_spin_orbitals=num_spin_orbitals, num_particles=num_particles, qubit_converter=converter,
                excitations='sd', initial_state=init_state)
    algorithm = VQE(ansatz, optimizer=opt, quantum_instance=backend, initial_point=parameters)

    result_cs = algorithm.compute_minimum_eigenvalue(qubit_op, aux_operators=[qubit_op_mdx, qubit_op_pdx])
    result_mdx = algorithm.compute_minimum_eigenvalue(qubit_op_mdx)
    result_pdx = algorithm.compute_minimum_eigenvalue(qubit_op_pdx)

    e = result_cs.eigenvalue + e_nr
    e_pdx = result_cs.aux_operator_eigenvalues[1] + e_nr_pdx
    e_mdx = result_cs.aux_operator_eigenvalues[0] + e_nr_mdx
    grad_cs = (e_pdx - e_mdx) / 2 / dx
    grad_brute = (result_pdx.eigenvalue + e_nr_pdx - result_mdx.eigenvalue - e_nr_mdx) / 2 / dx
    print("Energy", e)
    print("Gradient (CS):", grad_cs)
    print("Gradient (brute):", grad_brute)
    print("Gradient (FCI):", grad_fci)

    return e, e_pdx, e_mdx, e_fci, grad_cs[0], grad_fci, result_cs.optimal_point

# test
# r = 1.5 
# dx = 1e-3
# shots = 1

# compute_energy_and_grad(r, dx, shots)