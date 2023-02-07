import tequila as tq
import numpy as np
from tequila.objective.objective import Variable
import openfermion
from typing import Union
# this should be there already
# from vqe_utils import convert_PQH_to_tq_QH, convert_tq_QH_to_PQH,\
                      # fold_unitary_into_hamiltonian
from tequila.hamiltonian.folding import fold_unitary_into_hamiltonian

# not sure what to do with this
from energy_optimization import *
from do_annealing import *

import random # guess we could substitute that with numpy.random #TODO
import argparse
import pickle as pk


import matplotlib.pyplot as plt


# TODO
# |_| add explanation of stuff for everything

# Main hyperparameters
mu = 2.0         # lognormal dist mean
sigma = 0.4      # lognormal dist std
T_0 = 1.0        # T_0, initial starting temperature
# max_epochs = 25 # num_epochs, max number of iterations
max_epochs = 20 # num_epochs, max number of iterations
min_epochs = 3  # num_epochs, max number of iterations5
# min_epochs = 20  # num_epochs, max number of iterations
tol = 1e-6       # minimum change in energy required, otherwise terminate optimization
actions_ratio = [.15, .6, .25]
patience = 100
beta = 0.5
max_non_cliffords = 0
type_energy_eval='wfn'
num_population = 15
num_offsprings = 10
num_processors = 1

alpha = 0.9

# TODO
# |_| add docstring
def generate_hamiltonian_and_spa(geometry, basis, basis_type: str='pno', active_orbitals=None):
    '''
    <docstring>
    '''
    H, U = None, None

    if basis_type.lower() == 'pno':
        # here, basis is name that points to madness orbital data
        mol = tq.Molecule(name=basis, geometry=geometry, n_pno=None)
        lqm = mol.local_qubit_map(hcb=False)
        H = mol.make_hamiltonian().map_qubits(lqm).simplify()
    elif basis_type.lower() == 'gbs':
        mol = tq.Molecule(geometry=geometry, basis_set=basis,
                          active_orbitals=active_orbitals, backend='psi4')
        H = mol.make_hamiltonian().simplify()
        U = mol.make_uccsd_ansatz(trotter_steps=1)

    return H, U

# TODO
# don't know if wanna include ref energy computation at some point or not
# print("FCI ENERGY: ", mol.compute_energy(method="fci"))

# Set up molecules here, so 3x for each one
geometry = 'H 0. 0. 0.\n H 0. 0. 1.7'
name = 'H2'
basis = '6-31g'
# TODO |_| fill with actual stuff that can be computed
H, U_spa = generate_hamiltonian_and_spa(geometry=geometry, basis=basis, basis_type='gbs', active_orbitals=list(range(3)))
U = U_spa
n_qubits = H.n_qubits
# H, U_spa = generate_hamiltonian_and_spa(geometry, basis, basis_type: str='pno', active_orbitals=None)
type_energy_eval = 'wfn'

# TODO
# |_| update the stuff below (just copied)
# Clifford optimization in the context of reduced-size quantum circuits
starting_E = 123.456  # something arbitrary
reference = True  # whether to use reference energy within optimization
if reference:
    if type_energy_eval.lower() == 'wfn':
        starting_E, _ = minimize_energy(hamiltonian=H, n_qubits=n_qubits, type_energy_eval='wfn')
        print('starting energy wfn: {:.5f}'.format(starting_E), flush=True)
    elif type_energy_eval.lower() == 'qc':
        starting_E, _ = minimize_energy(hamiltonian=H, n_qubits=n_qubits, type_energy_eval='qc', cluster_circuit=U)
        print('starting energy spa: {:.5f}'.format(starting_E))
    else:
        raise Exception("type_energy_eval must be either 'wfn' or 'qc', but is", type_energy_eval)
    starting_E, _ = minimize_energy(hamiltonian=H, n_qubits=n_qubits, type_energy_eval='wfn')


if True:
    alphas = [.9]
    for alpha in alphas:
        print('Starting optimization, alpha = {:3f}'.format(alpha))

        # print('Energy to beat', minimize_energy(H, n_qubits, 'wfn')[0])
        simulated_annealing(hamiltonian=H, num_population=num_population,
                            num_offsprings=num_offsprings,
                            num_processors=num_processors,
                            tol=tol, max_epochs=max_epochs,
                            min_epochs=min_epochs, T_0=T_0, alpha=alpha,
                            actions_ratio=actions_ratio,
                            max_non_cliffords=max_non_cliffords,
                            verbose=True, patience=patience, beta=beta,
                            type_energy_eval=type_energy_eval.lower(),
                            cluster_circuit=U,
                            starting_energy=starting_E)



alter_cliffords = 2
if alter_cliffords == 1:
    print("starting to replace cliffords with non-clifford gates to see if that improves the current fitness")
    replace_cliff_with_non_cliff(hamiltonian=H, num_population=num_population,
                        num_offsprings=num_offsprings,
                        num_processors=num_processors,
                        tol=tol, max_epochs=max_epochs,
                        min_epochs=min_epochs, T_0=T_0, alpha=alphas[0],
                        actions_ratio=actions_ratio,
                        max_non_cliffords=max_non_cliffords,
                        verbose=True, patience=patience, beta=beta,
                        type_energy_eval=type_energy_eval.lower(),
                        cluster_circuit=U,
                        starting_energy=starting_E)
'''
elif alter_cliffords == 2:
    print("starting to replace cliffords with non-clifford gates to see if that improves the current fitness -- use 2")
    replace_2_cliff_with_non_cliff(hamiltonian=H, num_population=num_population,
                        num_offsprings=num_offsprings,
                        num_processors=num_processors,
                        tol=tol, max_epochs=max_epochs,
                        min_epochs=min_epochs, T_0=T_0, alpha=alphas[0],
                        actions_ratio=actions_ratio,
                        max_non_cliffords=max_non_cliffords,
                        verbose=True, patience=patience, beta=beta,
                        type_energy_eval=type_energy_eval.lower(),
                        cluster_circuit=U,
                        starting_energy=starting_E)
else:
    pass
'''
