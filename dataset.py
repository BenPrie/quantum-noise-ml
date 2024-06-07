from os import listdir, makedirs, path
from tqdm.notebook import tqdm
import numpy as np
import pickle
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# Helper function to filter the file list into the (non-split) files of only a given machine.
def filter_to_machine(file_list, machine):
    """
    Given a list of file paths and a machine, filter for only relevant paths for data extraction.

    :param file_list: List of file paths (as strings).
    :param machine: String machine title.
    :return: Filter object for the array of relevant file paths.
    """
    # List of words that specify different types of data (e.g. split).
    no_words = ['split', 'bis']

    return filter(
        lambda file: machine in file and not (any([word in file for word in no_words])), file_list
    )


def extract(machines, base_dir, extracted_dir, window_sizes, verbose=True):
    """
    'Extract' the executions of a given machine and translate into probability distributions over measurement outcomes.

    :param machines: List of machine names (as strings).
    :param base_dir: Directory (as a string) containing raw execution data.
    :param extracted_dir: Directory (as a string) for extracted execution and translated probability data to be saved to.
    :param window_sizes: List of "window sizes", defining windows in which to average probability distributions. A probabilities dataset will be created for each.
    :param verbose: Show progress bars.
    :return:
    """
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

                # Note: we will not be doing repeated measures, nor will we "read all bits" (whatever that means).

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


def create_dataset(machines, window_size, extracted_dir, dataset_path, file_name):
    """
    Collate the probability distributions for the given machines (in a given window) and write to compressed file.

    :param machines: List of machines (as strings) to include.
    :param window_size: Integer window size to look for in extracted files (using previous naming convention).
    :param extracted_dir: Directory (as a string) storing the probability distribution datasets for the machines.
    :param dataset_path: Directory to save dataset to.
    :param file_name: File name to use.
    :return: None.
    """
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
    np.savez_compressed(path.join(dataset_path, file_name), xs=xs, ys=ys)


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
    """
    Load and format a dataset over the given measurement steps from the data at the given path.
    The features are probability distributions (only), and the labels are indices for machines.

    Note that this has three different forms of return, depending on the passed arguments.

    :param dataset_path: Path (as a string) to the compressed dataset.
    :param steps: List of measurement steps (integers) to include in the dataset. If only a single step is needed, pass as a singleton array (the dataset shape will be simplified in this case).
    :param test_size: Float in [0, 1) indicating the proportion of the data to be held out for the test set.
    :param val_size: Float in [0, 1) indicating the proportion of the remaining training data to be held out for validation.
    :param shuffle: Whether the shuffle the samples in the dataset.
    :param as_torch: Whether to produce PyTorch return (i.e. Dataset or Dataloader object).
    :param batch_size: Batch size for use in Dataloader. If None (default), a Dataset object will be returned. Otherwise, a Dataloader object with the given batch size will be returned.
    :param seed: Seed for randomness.
    :return: Either of (A.) numpy arrays xs_train, xs_val, xs_test, ys_train, ys_val, ys_test / (B.) PyTorch Datasets train_dataset, val_dataset, test_dataset / (C.) PyTorch Dataloaders train_dataloader, val_dataloader, test_dataloader.
    """
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
        else: test_dataset = None

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


# Extend the Dataset class to offer machine indices too.
class ExtendedPyTorchDataset(Dataset):
    def __init__(self, xs: np.array, machine_indices: np.array, ys: np.array):
        self.xs = xs
        self.machine_indices = machine_indices
        self.ys = ys

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, idx):
        return self.xs[idx], self.machine_indices[idx], self.ys[idx]


# Reimagining the load_dataset function for the reverse task of classifying the measurement step.
def load_flipped_dataset(dataset_path, steps, test_size, val_size, shuffle=True, as_torch=True, batch_size=None, seed=42):
    # Setting randomness seeds.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Read in the dataset (expected .npz file with keys 'xs' and 'ys', as produced by create_dataset).
    dataset = np.load(dataset_path)
    xs_train, machine_indices_train = dataset['xs'][:, steps, :], dataset['ys']

    # Pull the measurement steps out of their machine groups to form one massive list of probability distributions.
    # i.e. if the original shape is (N, k, 2^n), then the new shape is (N * k, 2^n), but order is preserved (for now).
    xs_train = xs_train.reshape(-1, xs_train.shape[-1])

    # Measurement steps are in given order, repeated down the dataset.
    ys_train = np.array(list(steps) * machine_indices_train.shape[0])

    # Repeat the machine indices so that there is a machine label for each measurement step (rather than for th group).
    machine_indices_train = np.repeat(machine_indices_train, len(steps))

    if test_size > 0:
        # Train-test split.
        xs_train, xs_test, machine_indices_train, machine_indices_test, ys_train, ys_test = train_test_split(
            xs_train, machine_indices_train, ys_train, test_size=test_size, shuffle=shuffle
        )
    else:
        xs_test, machine_indices_test, ys_test = None, None, None

    if val_size > 0:
        # Train-val split.
        xs_train, xs_val, machine_indices_train, machine_indices_val, ys_train, ys_val = train_test_split(
            xs_train, machine_indices_train, ys_train, test_size=val_size, shuffle=shuffle
        )
    else:
        xs_val, machine_indices_val, ys_val = None, None, None

    # Special case has already been handled in reshaping (a nice bonus).

    # Convert to PyTorch if necessary.
    if as_torch:
        # Dataset objects.
        train_dataset = ExtendedPyTorchDataset(xs_train, machine_indices_train, ys_train)
        if val_size > 0: val_dataset = ExtendedPyTorchDataset(xs_val, machine_indices_val, ys_val)
        else: val_dataset = None
        if test_size > 0: test_dataset = ExtendedPyTorchDataset(xs_test, machine_indices_test, ys_test)
        else: test_dataset = None

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
    return xs_train, xs_val, xs_test, machine_indices_train, machine_indices_val, machine_indices_test, ys_train, ys_val, ys_test
