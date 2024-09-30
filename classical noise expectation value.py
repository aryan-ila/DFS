import numpy as np
import random
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from qiskit.quantum_info import Pauli, SparsePauliOp, DensityMatrix
from scipy.linalg import expm
from statistics import mean

# Create a quantum circuit
n = 1
Qc = QuantumCircuit(n, n)
Qc.barrier()

# Density matrix of the quantum state
ρ_AB = qi.DensityMatrix.from_instruction(Qc)

# Define observable (Pauli Z operator)
observable = Pauli("Z")

# Pauli operators for creating Hamiltonians
observable_1 = SparsePauliOp("X")
observable_3 = SparsePauliOp("Z")
H_P = observable_1 + observable_3

# Empty lists for storing time and expectation values
time = np.arange(80, 100, 0.01)
EV_NL = []
EV_A_avg = []
EV_B_avg = []

# Step 1: Compute noiseless expectation values (outside the loop over a_i)
for t in time:
    # Unitary evolution with noiseless Hamiltonian
    U_1 = expm(-1j * H_P * t)
    ρ_AB1 = ρ_AB.evolve(U_1)
    
    # Noiseless expectation value
    expectation_value_1 = ρ_AB1.expectation_value(observable)
    EV_NL.append(expectation_value_1.real)

# Step 2: Loop over 10 different values of a_i to compute noisy expectation values
num_samples = 10
for i in range(num_samples):
    a_i = random.uniform(0, 2 * np.pi) / 10
    
    # Lists for storing noisy expectation values for the current a_i
    EV_A = []
    EV_B = []
    
    for t in time:
        # Creating noisy Hamiltonians
        observable_6 = a_i * SparsePauliOp("Y")
        observable_7 = (np.cos(a_i) - 1) * SparsePauliOp("X") + np.sin(a_i) * SparsePauliOp("Y")
        
        H = observable_1 + observable_3 + observable_6
        HH = observable_1 + observable_3 + observable_7
        
        # Unitaries for the evolution of the system with noise
        U_2 = expm(-1j * H * t)
        U_3 = expm(-1j * HH * t)
        
        # Evolved density matrices for noisy cases
        ρ_AB2 = ρ_AB.evolve(U_2)
        ρ_AB3 = ρ_AB.evolve(U_3)
        
        # Noisy expectation values
        expectation_value_2 = ρ_AB2.expectation_value(observable)
        expectation_value_3 = ρ_AB3.expectation_value(observable)
        
        EV_A.append(expectation_value_2.real)
        EV_B.append(expectation_value_3.real)
    
    # After computing for the current a_i, average over all noisy samples
    if i == 0:
        # Initialize the lists for averaging across all samples
        EV_A_avg = np.array(EV_A)
        EV_B_avg = np.array(EV_B)
    else:
        # Accumulate results for averaging
        EV_A_avg += np.array(EV_A)
        EV_B_avg += np.array(EV_B)

# Step 3: Average the noisy results over all the a_i samples
EV_A_avg /= num_samples
EV_B_avg /= num_samples

# Plot the results
plt.plot(time, EV_NL, label='Noiseless Expectation value')
plt.plot(time, EV_A_avg, label='Expectation value with weak classical noise (Averaged)')
plt.plot(time, EV_B_avg, label='Expectation value without weak classical noise (Averaged)')

# Graph labeling and grid
plt.xlabel("time(s)")
plt.ylabel(r"$\langle z \rangle$")
plt.grid()
plt.legend()
plt.show()
