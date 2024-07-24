# Imports, as always
from os import makedirs, path
import pandas as pd
from typing import List
from datetime import date

# Circuitry.
from qiskit.circuit import QuantumCircuit

# Runtime.
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService

# Scripts.
from scipts.utils import reset_seed
from scipts import circuit


# Runtime service. NOT TO BE EXPOSED PUBLICLY.
service = QiskitRuntimeService(channel='ibm_quantum', token='90f74dbca51821d1203799d5afc2a9c70b836eb1a6377e53367bb9896ec9e7faa9678a9981ad6bf6bf44dd5accb83ee576a3e6eb8d85249f7d98a138f2d8a5f1')

# Data save path.
data_save_path = './data/fake-backend'
makedirs(data_save_path, exist_ok=True)


def measurement_step_routine(brickwork_circuit: QuantumCircuit, backend, k: int, n_runs: int, n_shots: int = 1000,
                             write_to_file: bool = False, seed=None) -> pd.DataFrame:
    # RNG.
    reset_seed(seed)

    # Chained circuit.
    chained_circuit = circuit.build_chained_circuit(brickwork_circuit, k, parameterised_input=False)

    # Infer circuit configuration.
    n, d = brickwork_circuit.num_qubits, brickwork_circuit.depth()

    # Transpile.
    transpiled_circuit = transpile(chained_circuit, backend=backend)

    # Sampler.
    sampler = Sampler(mode=backend)

    # Dataframe to hold results.
    results_df = pd.DataFrame(
        columns=['n', 'd', 'step', 'k', 'run', 'n_shots', 'device'] + [format(i, f'0{n}b') for i in range(2 ** n)])

    if write_to_file:
        # Create a subdirectory for the backend if none exists.
        makedirs(path.join(data_save_path, backend.backend_name), exist_ok=True)

        # File name.
        file_path = path.join(
            data_save_path, backend.backend_name, date.today().strftime('measurement-step-(%Y-%m-%d).csv')
        )

        # Empty file.
        results_df.to_csv(file_path, index=False, header=True)

    for run in range(1, n_runs + 1):
        # Run (via the sampler).
        job = sampler.run([(transpiled_circuit, None, n_shots)])

        # Convert into outcome distributions for each measurement step.
        outcomes = {
            step: {
                key: value / n_shots
                for key, value in job.result()[0].data[f'step{step}'].get_counts().items()
            } for step in range(k + 1)
        }

        # Compact into a DataFrame object.
        for step, outcome_distribution in outcomes.items():
            run_result_df = pd.DataFrame(
                columns=['n', 'd', 'step', 'k', 'run', 'n_shots', 'device'] + [format(i, f'0{n}b') for i in
                                                                               range(2 ** n)])

            # Translate the entries of the outcome distribution into the DataFrame.
            # Any outcome states not measured will be left as NaN in this DataFrame.
            for state, prob in outcome_distribution.items():
                run_result_df[state] = [prob]

            # "Meta" data.
            run_result_df['n'] = [n]
            run_result_df['d'] = [d]
            run_result_df['step'] = [step]
            run_result_df['k'] = [k]
            run_result_df['run'] = [run]
            run_result_df['n_shots'] = [n_shots]
            run_result_df['device'] = [backend.backend_name]

            # Add to the overall results.
            results_df = pd.concat([results_df, run_result_df], ignore_index=True)

            # Write to file.
            if write_to_file:
                run_result_df.to_csv(file_path, mode='a', index=False, header=False)

    return results_df


def depth_varied_routine(n: int, ds: List[int], backend, n_runs: int, n_shots: int = 1000, write_to_file: bool = False,
                         seed=None) -> pd.DataFrame:
    # RNG.
    reset_seed(seed)

    # Dataframe to hold results.
    results_df = pd.DataFrame(
        columns=['n', 'd', 'run', 'n_shots', 'device'] + [format(i, f'0{n}b') for i in range(2 ** n)]
    )

    if write_to_file:
        # Create a subdirectory for the backend if none exists.
        makedirs(path.join(data_save_path, backend.backend_name), exist_ok=True)

        # File name.
        file_path = path.join(
            data_save_path, backend.backend_name, date.today().strftime('depth-varied-(%Y-%m-%d).csv')
        )

        # Empty file.
        results_df.to_csv(file_path, index=False, header=True)

    for d in ds:
        # Build the depth-d brickwork circuit.
        brickwork_circuit = circuit.generate_brickwork_circuit(n, d, seed=seed)

        # Build the full circuit.
        experiment_circuit = circuit.build_experiment_circuit(n, d, brickwork_circuit)

        # Transpile.
        transpiled_circuit = transpile(experiment_circuit, backend=backend)

        # Sampler.
        sampler = Sampler(mode=backend)

        for run in range(1, n_runs + 1):
            # Run (via the sampler).
            job = sampler.run([(transpiled_circuit, None, n_shots)])

            # Convert into an outcome distribution.
            outcomes = {
              key: value / n_shots
              for key, value in job.result()[0].data.meas.get_counts().items()
            }

            # Compact into a DataFrame object.
            run_result_df = pd.DataFrame(
                columns=['n', 'd', 'run', 'n_shots', 'device'] + [format(i, f'0{n}b') for i in range(2 ** n)]
            )

            # Translate the entries of the outcome distribution into the DataFrame.
            # Any outcome states not measured will be left as NaN in this DataFrame.
            for state, prob in outcomes.items():
                run_result_df[state] = [prob]

            # "Meta" data.
            run_result_df['n'] = [n]
            run_result_df['d'] = [d]
            run_result_df['run'] = [run]
            run_result_df['n_shots'] = [n_shots]
            run_result_df['device'] = [backend.backend_name]

            # Add to the overall results.
            results_df = pd.concat([results_df, run_result_df], ignore_index=True)

            # Write to file.
            if write_to_file:
                run_result_df.to_csv(file_path, mode='a', index=False, header=False)

    return results_df


def identity_circuit_routine():
    # We probably won't be doing this.
    pass
