import numpy as np
from torch.utils.data import Dataset
import os
from dotenv import load_dotenv
load_dotenv()
main_dir = os.getenv("DATA_PATH")

class BinaryDataset(Dataset):
    def __init__(self, file_path, win_length, positions, win_shift=0, dtype=np.uint32):
        """
        Initializes the dataset.

        Args:
            file_path (str): Path to the binary file.
            win_length (int): Length of each data window to extract.
            positions (list or np.array): List of starting positions in the binary file for windows.
            win_shift (int): Optional shift applied to window start positions (default: 0).
            dtype (numpy dtype): Data type for reading binary data (default: np.uint32).
        """
        self.file_path = file_path
        self.win_length = int(win_length)
        self.positions = positions
        self.win_shift = int(win_shift)
        self.dtype = dtype

        # Memory-map the binary file to allow efficient reading without loading the entire file into memory.
        # Skips the first 4 bytes of the file using offset=4 (often used to skip a header or metadata).
        self.file_map = np.memmap(self.file_path, dtype=dtype, mode='r', offset=4)

    def load_and_process_window(self, file_map, start):
        """
        Loads a window of data from the memory-mapped file and applies processing.

        Args:
            file_map (np.memmap): Memory-mapped binary file.
            start (int): Starting index for the window.

        Returns:
            np.array: Processed data window.
        """
        # Extract a slice of raw data of length `win_length` starting at `start`
        np_data = np.array(file_map[start:start + self.win_length], dtype=self.dtype)

        # Bit-shift each 32-bit value to the right by 8 bits, convert to float64,
        # normalize from raw values to a range between -10.24 and +10.24
        np_data = ((np.right_shift(np_data, 8).astype(np.float64) / (2 ** 23)) * 10.24 - 10.24)

        return np_data

    def __len__(self):
        """
        Returns the total number of data windows in the dataset.
        """
        return len(self.positions)

    def __getitem__(self, index):
        """
        Retrieves and processes a single data window based on the index.

        Args:
            index (int): Index into the list of positions.

        Returns:
            np.array: Processed data window.
        """
        # Calculate the actual start index for the window
        start = self.positions[index] - self.win_shift

        # Load and process the data window
        data = self.load_and_process_window(self.file_map, start)

        return data


def create_data_sets(channel, win_length=2000, shift=0, pos_prefix="_bis", file_sufixe="", differenciate=True):
    """
    Creates datasets for single and pileup data, along with additional processed information.

    Args:
        channel (int): Channel number to process.
        win_length (int, optional): Length of each data window. Defaults to 2000.
        shift (int, optional): Shift applied to window start positions. Defaults to 0.
        pos_prefix (str, optional): Prefix for position files. Defaults to "_bis".
        file_sufixe (str, optional): Suffix for binary files. Defaults to "".
        differenciate (bool, optional): Whether to use differentiated data paths. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - data_single (BinaryDataset): Dataset for single data.
            - data_pileup (BinaryDataset): Dataset for pileup data.
            - dt_pileup (numpy.ndarray): Time differences for pileup data.
            - fracE_pileup (numpy.ndarray): Fractional energy for pileup data.
            - E_pileups (numpy.ndarray): Energy values for pileup data.
    """
    partial = 0
    dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
    dataset_single_name = "pup_n1-d0_000813_20230628T161508"

    # Construct file paths for pileup data
    base_path = f"/channel{channel}/"
    dif_path = "dif/" if differenciate else ""
    file_path_pileup = f"{main_dir}{base_path}{dif_path}{dataset_pileup_name}_{channel:03}_{partial:03}_dif.bin{file_sufixe}"
    path_pos_pileup = f"{main_dir}{base_path}{dataset_pileup_name}_{channel:03}_{partial:03}{pos_prefix}.pos"

    # Load pileup positions and process
    pos_file_pileup = np.loadtxt(path_pos_pileup)[:15000]
    dt_pileup = pos_file_pileup[:, -1]
    fracE_pileup = pos_file_pileup[:, 2] / pos_file_pileup[:, 1]
    E_pileups = pos_file_pileup[:, 1:3]
    dt_pileup[fracE_pileup > 1] = -dt_pileup[fracE_pileup > 1]
    fracE_pileup[fracE_pileup > 1] = 1 / fracE_pileup[fracE_pileup > 1]
    positions_pileup = (pos_file_pileup[:, 0] // 4).astype(int)

    # Create BinaryDataset for pileup data
    data_pileup = BinaryDataset(file_path_pileup, win_length, positions_pileup, win_shift=win_length // 2 - shift)

    # Construct file paths for single data
    file_path_single = f"{main_dir}{base_path}{dif_path}{dataset_single_name}_{channel:03}_{partial:03}_dif.bin{file_sufixe}"
    path_pos_single = f"{main_dir}{base_path}{dataset_single_name}_{channel:03}_{partial:03}{pos_prefix}.pos"

    # Load single positions and create BinaryDataset
    pos_file_single = np.loadtxt(path_pos_single)[:15000]
    positions_single = (pos_file_single[:, 0] // 4).astype(int)
    data_single = BinaryDataset(file_path_single, win_length, positions_single, win_shift=win_length // 2 - shift)

    return data_single, data_pileup, dt_pileup, fracE_pileup, E_pileups

def get_channel_specs(channel, N_deriv=1):
    """
    Loads and processes filter data for a given channel.

    Args:
        channel (int): Channel number to process.
        N_deriv (int, optional): Number of derivatives to compute. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - H_unit (numpy.ndarray): Normalized filter response.
            - S (numpy.ndarray): Fourier transform of the differentiated mean pulse.
            - bslspec (numpy.ndarray): Baseline spectrum.
            - w (numpy.ndarray): frequency array.
    """
    # Load baseline spectrum
    bslspec = np.fromfile(f"{main_dir}/channel{channel}/dif/raw_noise_d{N_deriv}.bin_spec.bin")
    # Load mean pulse, trying two possible file paths
    try:
        meanpulse = np.fromfile(f"{main_dir}/injected_pulses/000813_20230628T161508_{channel:03}_000.bin_edmean.bin")
    except FileNotFoundError:
        meanpulse = np.fromfile(f"{main_dir}/injected_pulses/000813_20230628T161508_{channel:03}_001.bin_edmean.bin")

    # Compute the N-th derivative of the mean pulse
    dmeanpulse = np.diff(meanpulse, n=N_deriv, prepend=[0] * N_deriv)
    size = len(bslspec)
    dmeanpulse = dmeanpulse[np.argmax(dmeanpulse)-size//2 : np.argmax(dmeanpulse)+size//2]
    if len(dmeanpulse) < size:
        dmeanpulse = np.pad(dmeanpulse, (0, size - len(dmeanpulse)), 'constant')
    dmeanpulse /= np.max(np.abs(dmeanpulse))
    # Perform Fourier transform and compute angular frequency
    S = np.fft.fft(dmeanpulse)
    freq = np.fft.fftfreq(len(S), 1e-4)
    w = 2 * np.pi * freq

    # Compute filter response and normalize it
    H = np.conj(S) / bslspec
    H_unit = H / np.mean(np.abs(H * S))

    return H_unit, S, bslspec, w