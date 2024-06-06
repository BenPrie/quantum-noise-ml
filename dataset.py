from os import listdir, makedirs, path
from tqdm.notebook import tqdm
import numpy as np
import pickle
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# Helper function to filter the file list into the (non-split) files of only a given machine.
def filter_to_machine(file_list, machine):
    # List of words that specify different types of data (e.g. split).
    no_words = ['split', 'bis']

    return filter(
        lambda file: machine in file and not (any([word in file for word in no_words])), file_list
    )


def extract(machines, base_dir, extracted_dir, window_sizes, verbose=True):
    # List of files in the base path.
    file_list = listdir(base_dir)

    # Generate the output path.
    makedirs(extracted_dir, exist_ok=True)

    # Extract executions for each machine.
    for machine in (tqdm(machines, desc='Extracting executions') if verbose else machines):
        executions = []

        # For each file belonging to the current machine.
        for file in tqdm(filter_to_machine(file_list, machine), desc='Reading files', total=250):
            # Read the contents of the file.
            contents = pickle.load(open(path.join(base_dir, file), 'rb'))

            # For each run of the circuit (of which there are 8000 in the Martina paper and data).
            for n in range(len(contents['results'][0]['data']['memory'])):
                current_execution = []

                # Note: we will not be doing repeated measures, nor will we "read all bits".

                # For each measurement step t (of which there are 9 in the Martina paper and data).
                for t in range(len(contents['results'])):
                    execution = int(contents['results'][t]['data']['memory'][n], 0)
                    current_execution.append(execution)

                executions.append(current_execution)

        # Cast to numpy array.
        executions = np.array(executions)

        # Save the full executions of this machine.
        np.savetxt(path.join(extracted_dir, f'{machine}-executions.csv'), executions)

        # Averaging probabilities over window(s).
        for window_size in window_sizes:
            # The window size must cleanly divide the number of executions.
            if executions.shape[0] % window_size != 0: raise (Exception('Indivisible by window size.'))

            # Initialise probabilities array.
            probs = np.zeros(shape=(
                executions.shape[0] // window_size, executions.shape[1], np.unique(executions).shape[0]
            ), dtype=np.float32)

            # Calculate probabilities with the given window size.
            for n in tqdm(range(executions.shape[0]), desc='Calculating probabilities'):
                i = n // window_size

                for t in range(executions.shape[1]):
                    probs[i, t, executions[n, t]] += 1

            probs = probs / window_size

            # Save the window.
            np.save(path.join(extracted_dir, f'{machine}-probabilities-{window_size}.npy'), probs)


def create_dataset(machines, extracted_dir, dataset_path, window_size):
    # Generate the output directory.
    makedirs(dataset_path, exist_ok=True)

    # Pack probability distributions from all machines into a single array. Keep order here incase we want shuffling.
    probs = [np.load(path.join(extracted_dir, f'{machine}-probabilities-{window_size}.npy')) for machine in machines]
    order = [np.arange(prob.shape[0]) for prob in probs]

    # Convert to the feature (x) label (y) format.
    xs, ys = []
    for i in range(min(map(len, order))):
        for p in range(len(probs)):
            xs.append(probs[p][order[p][i]])
            ys.append(p)

    # Numpify those arrays.
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    # Save in this format.
    np.savez_compressed(path.join(dataset_path, f'all-dataset-{window_size}'), xs=xs, ys=ys)


# PyTorch Dataset class for handling these data. This isn't really necessary, I just like it.
class PyTorchDataset(Dataset):
    def __init__(self, xs: np.array, ys: np.array):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


def load_dataset(dataset_path, steps, test_size, val_size, shuffle=True, as_torch=True, batch_size=None, seed=42):
    # Setting randomness seeds.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Read in the dataset (expected .npz file with keys 'xs' and 'ys', as produced by create_dataset).
    dataset = np.load(dataset_path)

    if test_size > 0:
        # Train-test split the data.
        xs_train, xs_test, ys_train, ys_test = train_test_split(dataset['xs'][:, steps, :], dataset['ys'], test_size=test_size, shuffle=shuffle)
    else:
        xs_train, ys_train = dataset['xs'][:, steps, :], dataset['ys']
        xs_test, ys_test = None, None

    if val_size > 0:
        # Train-val split the data.
        xs_train, xs_val, ys_train, ys_val = train_test_split(xs_train, ys_train, test_size=val_size, shuffle=shuffle)
    else:
        xs_val, ys_val = None, None

    # Special case for single-step data to be handled (unnecessary nesting of arrays).
    if len(steps) == 1:
        xs_train = xs_train.reshape((xs_train.shape[0], xs_train.shape[-1]))
        if val_size > 0: xs_val = xs_val.reshape((xs_val.shape[0], xs_val.shape[-1]))
        if test_size > 0: xs_test = xs_test.reshape((xs_test.shape[0], xs_test.shape[-1]))

    # Convert to PyTorch if necessary.
    if as_torch:
        # Dataset objects.
        train_dataset = PyTorchDataset(xs_train, ys_train)
        if val_size > 0: val_dataset = PyTorchDataset(xs_val, ys_val)
        else: val_dataset = None
        if test_size > 0: test_dataset = PyTorchDataset(xs_test, ys_test)

        # Convert further into Dataloaders if a batch size is specified.
        if batch_size:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            if val_size > 0: val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
            else: val_dataloader = None
            if test_size > 0: test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
            else: test_dataloader = None

            return train_dataloader, val_dataloader, test_dataloader

        # Otherwise, a dataset will do.
        return train_dataset, val_dataset, test_dataset

    # Otherwise, return the numpy arrays.
    return xs_train, xs_val, xs_test, ys_train, ys_val, ys_test
