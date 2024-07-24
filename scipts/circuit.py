# Imports, as always
import numpy as np

# Circuitry.
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary

# Scripts.
from scipts.utils import reset_seed


def generate_brickwork_circuit(n: int, d: int, seed=None):
    # A brickwork circuit only really makes sense with at least 3 qubits.
    assert n > 2, 'Brickwork architectures require at least 3 qubits to be worth the bother.'

    # RNG.
    reset_seed(seed)

    # Instantiate the circuit object.
    circuit = QuantumCircuit(n)

    for l in range(d):
        # Even layer arrangement.
        if l % 2 == 0:
            # Loop over the qubits onto to which the 'bricks' will be 'layed'.
            for i in range(n // 2):
                # Add a Haar random unitary from U(4).
                circuit.append(
                    UnitaryGate(random_unitary(4), label='$U_{' + str(l) + str(i) + '}$'),
                    [2*i, 2*i + 1]
                )

        # Odd layer arrangement is offset by one qubit.
        else:
            # Looping again.
            for i in range(1, n // 2 + 1):
                # Add again.
                circuit.append(
                    UnitaryGate(random_unitary(4), label='$U_{' + str(l) + str(i) + '}$'),
                    [2*i - 1, 2*i]
                )

    return circuit


def generate_parameterised_input_layer(n: int):
    # Instantiate a circuit.
    circuit = QuantumCircuit(n)

    # Parameterised rotation on each qubit.
    input_parameters = ParameterVector(name='in', length=n)
    for i in range(n):
        circuit.rx(np.pi * input_parameters[i], i)

    return circuit


def build_experiment_circuit(n: int, d: int, brickwork_circuit=None, parameterised: bool = False, seed=None):
    # Quantum circuit.
    q_reg = QuantumRegister(n, name='q')
    c_reg = ClassicalRegister(n, name='meas')

    # Circuit object.
    circuit = QuantumCircuit(q_reg, c_reg)

    # Preparation layer.
    if parameterised:
        circuit.compose(generate_parameterised_input_layer(n), inplace=True)
    else:
        circuit.h(range(n))
        circuit.measure(q_reg, c_reg)

    circuit.barrier()

    # Brickwork layer.
    if brickwork_circuit is None:
        circuit.compose(generate_brickwork_circuit(n, d, seed=seed), inplace=True)
    else:
        circuit.compose(brickwork_circuit, inplace=True)

    circuit.barrier()

    # Measurement layer.
    circuit.measure(q_reg, c_reg)

    return circuit


def build_chained_circuit(brickwork_circuit: QuantumCircuit, k: int, parameterised_input: bool = False):
    assert k > 1, 'Number of measurement steps (k) must be a positive integer.'

    # Quantum circuit.
    n = brickwork_circuit.num_qubits
    q_reg = QuantumRegister(n, name='q')
    c_regs = [ClassicalRegister(n, name=f'step{i}') for i in range(k + 1)]

    # Circuit object.
    circuit = QuantumCircuit(*([q_reg] + c_regs))

    # Set up an input layer.
    if parameterised_input:
        input_layer = generate_parameterised_input_layer(n)
        circuit.compose(input_layer, inplace=True)
    else:
        circuit.h(range(n))
        circuit.measure(q_reg, c_regs[0])

    # Alternate between brickwork and measurement for k steps.
    for i in range(1, k + 1):
        circuit.barrier()
        circuit.compose(brickwork_circuit, inplace=True)
        circuit.barrier()
        circuit.measure(q_reg, c_regs[i])

    return circuit


def build_identity_circuit(n: int, d: int, brickwork_circuit=None, parameterised: bool = False, seed=None):
    # Quantum circuit.
    q_reg = QuantumRegister(n, name='q')
    c_reg = ClassicalRegister(n, name='meas')

    # Circuit object.
    circuit = QuantumCircuit(q_reg, c_reg)

    # Prepare the input.
    if parameterised:
        circuit.compose(generate_parameterised_input_layer(n), inplace=True)
    else:
        circuit.h(range(n))
        circuit.measure(q_reg, c_reg)

    circuit.barrier()

    # Add a brickwork circuit.
    if brickwork_circuit is None:
        brickwork_circuit = generate_brickwork_circuit(n, d, seed=seed)
    circuit.compose(brickwork_circuit, inplace=True)

    circuit.barrier()

    # Add it's inverse.
    circuit.compose(brickwork_circuit.inverse(), inplace=True)

    circuit.barrier()

    # Measure.
    circuit.measure(q_reg, c_reg)

    return circuit

