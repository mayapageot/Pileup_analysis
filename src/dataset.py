import numpy as np
from torch.utils.data import Dataset
import os
import glob
from dotenv import load_dotenv
import ast
import torch
load_dotenv()
main_dir = os.getenv("DATA_PATH")
# if os.path.isdir(main_dir) == False:
#     main_dir = "/local/home/mp274748/Documents/data/"


class NumpyDataset(Dataset):
    def __init__(self, np_array):
        self.data = torch.from_numpy(np_array)  # convert to torch.Tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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


class BinaryDatasetDenoised(BinaryDataset):
    def __init__(self, file_path, denoised_data, win_length, positions, win_shift=0, dtype=np.uint32):
        super().__init__(file_path, win_length, positions, win_shift, dtype)
        self.denoised_data = denoised_data

    def __getitem__(self, index):
        start = self.positions[index] - self.win_shift
        raw_data = self.load_and_process_window(self.file_map, start) - self.denoised_data[index]
        return raw_data

def find_file(target,specific_subdir=""):
    file_path = glob.glob(os.path.join(main_dir, specific_subdir,"**", target), recursive = True)[0]
    return file_path


def find_files(dataset_name, channel, pos_prefix="_bis", differenciate=True,file_suffix="",specific_subdir=""):
    modif = "_dif" if differenciate else ""
    target = dataset_name + f"_{channel:03}_???{modif}.bin{file_suffix}"
    file_path = find_file(target, specific_subdir)
    target_pos = dataset_name + f"_{channel:03}_???{pos_prefix}.pos"
    path_pos = find_file(target_pos, specific_subdir)
    return file_path,path_pos



def create_data_sets(channel, win_length=2000, shift=0, pos_prefix="_bis", file_suffix= "", differenciate=False,
                     specific_subdir="RUN9_pulse_injected_new/"):
    """
    Creates datasets for single and pileup data, along with additional processed information.

    Args:
        channel (int): Channel number to process.
        win_length (int, optional): Length of each data window. Defaults to 2000.
        shift (int, optional): Shift applied to window start positions. Defaults to 0.
        pos_prefix (str, optional): Prefix for position files. Defaults to "_bis".
        file_suffix (str, optional): Suffix for binary files. Defaults to "".
        differenciate (bool, optional): Whether to use differentiated data paths. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - data_single (BinaryDataset): Dataset for single data.
            - data_pileup (BinaryDataset): Dataset for pileup data.
            - dt_pileup (numpy.ndarray): Time differences for pileup data.
            - fracE_pileup (numpy.ndarray): Fractional energy for pileup data.
            - e_pileups (numpy.ndarray): Energy values for pileup data.
    """
    dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
    file_path_pileup,path_pos_pileup = find_files(dataset_pileup_name, channel,
                                                  pos_prefix = pos_prefix, differenciate = differenciate,
                                                  file_suffix=file_suffix,
                                                  specific_subdir=specific_subdir)
    # Load pileup positions and process
    pos_file_pileup = np.loadtxt(path_pos_pileup)[:15000]
    dt_pileup = pos_file_pileup[:, -1]
    r_pileup = pos_file_pileup[:, 2]/(pos_file_pileup[:, 2] + pos_file_pileup[:, 1])
    e_pileups = pos_file_pileup[:, 1:3]
    positions_pileup = (pos_file_pileup[:, 0] // 4).astype(int)

    # Create BinaryDataset for pileup data
    data_pileup = BinaryDataset(file_path_pileup, win_length, positions_pileup, win_shift=win_length // 2 - shift)

    dataset_single_name = "pup_n1-d0_000813_20230628T161508"
    file_path_single, path_pos_single = find_files(dataset_single_name, channel,
                                                  pos_prefix = pos_prefix, differenciate = differenciate,
                                                  file_suffix=file_suffix,
                                                  specific_subdir=specific_subdir)
    # Load single positions and create BinaryDataset
    pos_file_single = np.loadtxt(path_pos_single)[:15000]
    positions_single = (pos_file_single[:, 0] // 4).astype(int)
    data_single = BinaryDataset(file_path_single, win_length, positions_single, win_shift=win_length // 2 - shift)

    return data_single, data_pileup, dt_pileup, r_pileup, e_pileups

def create_data_sets_denoise(channel, win_length=2000, shift=0, pos_prefix="_bis", file_suffix= "", differenciate=False,
                     specific_subdir="RUN9_pulse_injected_new/"):
    """
    Creates datasets for single and pileup data, along with additional processed information.

    Args:
        channel (int): Channel number to process.
        win_length (int, optional): Length of each data window. Defaults to 2000.
        shift (int, optional): Shift applied to window start positions. Defaults to 0.
        pos_prefix (str, optional): Prefix for position files. Defaults to "_bis".
        file_suffix (str, optional): Suffix for binary files. Defaults to "".
        differenciate (bool, optional): Whether to use differentiated data paths. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - data_single (BinaryDataset): Dataset for single data.
            - data_pileup (BinaryDataset): Dataset for pileup data.
            - dt_pileup (numpy.ndarray): Time differences for pileup data.
            - fracE_pileup (numpy.ndarray): Fractional energy for pileup data.
            - e_pileups (numpy.ndarray): Energy values for pileup data.
    """
    dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
    file_path_pileup,_ = find_files(dataset_pileup_name, channel,
                                                  pos_prefix = pos_prefix, differenciate = differenciate,
                                                  file_suffix=file_suffix,
                                                  specific_subdir=specific_subdir)
    # Load pileup positions and process
    channels = [3,5,9,11,13,15,17,19]
    Global_stds = [0.00021900556748732924, 0.00028078060131520033, 0.00047128828009590507, 0.0003684174735099077,
                   0.0005077228997834027, 0.0004087088746018708, 0.0002996263501700014, 0.0011633249232545495]
    positions_pileup = torch.load(f"{main_dir}/RUN9/noise_file_RUN9/pup_n1-d8_000813_20230628T161508_{channel:03}_000_bis.pt").numpy()
    noise_pred_pileup = np.load(f"{main_dir}/RUN9/noise_file_RUN9/pulses_pred_d8_{channel:03}.npy")
    # Create BinaryDataset for pileup data
    data_pileup = BinaryDatasetDenoised(file_path_pileup,
                                        noise_pred_pileup[:,noise_pred_pileup.shape[1] // 2 - win_length // 2 - shift:
                                                     noise_pred_pileup.shape[1] // 2 + win_length // 2 - shift] *
                                        Global_stds[channels.index(channel)],
                                        win_length, positions_pileup,
                                        win_shift = win_length // 2 - shift)
    dataset_single_name = "pup_n1-d0_000813_20230628T161508"
    file_path_single, path_pos_single = find_files(dataset_single_name, channel,
                                                  pos_prefix = pos_prefix, differenciate = differenciate,
                                                  file_suffix=file_suffix,
                                                  specific_subdir=specific_subdir)
    # Load single positions and create BinaryDataset
    positions_single = torch.load(f"{main_dir}/RUN9/noise_file_RUN9/pup_n1-d0_000813_20230628T161508_{channel:03}_000_bis.pt").numpy()
    noise_pred_single= np.load(f"{main_dir}/RUN9/noise_file_RUN9/pulses_pred_d0_{channel:03}.npy")
    data_single = BinaryDatasetDenoised(file_path_single,
                                        noise_pred_single[:, noise_pred_single.shape[1] // 2 - win_length // 2 - shift:
                                                             noise_pred_single.shape[
                                                                 1] // 2 + win_length // 2 - shift] *
                                        Global_stds[channels.index(channel)],
                                        win_length, positions_single,
                                        win_shift = win_length // 2 - shift)
    return data_single, data_pileup

def get_channel_specs(channel, n_deriv=1, window_fct=np.ones,nps_name="raw_noise"):
    """
    Loads and processes filter data for a given channel.

    Args:
        channel (int): Channel number to process.
        n_deriv (int, optional): Number of derivatives to compute. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - H_unit (numpy.ndarray): Normalized filter response.
            - S (numpy.ndarray): Fourier transform of the differentiated mean pulse.
            - nps (numpy.ndarray): Baseline spectrum.
            - w (numpy.ndarray): frequency array.
    """
    # Load baseline spectrum
    meas_name = "000813_20230628T161508"

    nps = np.fromfile(find_file(f"{nps_name}_d{n_deriv}.bin_spec.bin",
                                       specific_subdir = f"RUN9_pulse_injected_new/channel{channel}"))
    meanpulse = np.fromfile(find_file(f"{meas_name}_{channel:03}_???.bin_edmean.bin",
                                       specific_subdir = "RUN9_pulse_injected_new/"))
    meanpulse /= np.max(np.abs(meanpulse))
    # Compute the N-th derivative of the mean pulse
    dmeanpulse = np.diff(meanpulse, n=n_deriv, prepend= [0] * n_deriv)
    pulse_factor = np.max(np.abs(dmeanpulse))
    size = len(nps)
    dmeanpulse = dmeanpulse[np.argmax(dmeanpulse)-size//2: np.argmax(dmeanpulse)+size//2]
    if len(dmeanpulse) < size:
        dmeanpulse = np.pad(dmeanpulse, (0, size - len(dmeanpulse)), 'constant')
    dmeanpulse /= np.max(np.abs(dmeanpulse))
    # Perform Fourier transform and compute angular frequency
    S = np.fft.fft(dmeanpulse*window_fct(size))
    freq = np.fft.fftfreq(size, 1e-4)
    w = 2 * np.pi * freq

    # Compute filter response and normalize it
    H = np.conj(S) / nps
    H_unit = H / np.mean(np.abs(H * S))

    return H_unit, S, nps, w, pulse_factor


def load_params_simu(filename):
    params = {}
    with open(filename, "r") as f:
        for line in f:
            # Remove inline comments
            line = line.split("#", 1)[0].strip()
            # Skip empty lines
            if not line:
                continue
            # Parse key=value
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().rstrip(";")
                # Try to parse Python literal
                try:
                    parsed_value = ast.literal_eval(value)
                except:
                    parsed_value = value  # leave as string

                params[key] = parsed_value

    return params

def get_amp_Q_val(channel):
    target = f"000813_20230628T161508_{channel:03}_???-pup_single.injconf"
    params = load_params_simu(find_file(target, specific_subdir = "RUN9_pulse_injected_new/"))
    return params["LY_close"] * 3.034 * params["ampl_Tl"] / 2.614511 / params["LY_far"] * params["area_factor"]

def get_noise_dataset(channel,denoise=False):
    win_length = 2000
    shift = 0
    if denoise :
        positions_single = torch.load(f"/media/mp274748/Transcend/data/LSC/RUN9/noise_file_RUN9/pup_n1-d0_000813_20230628T161508_{channel:03}_000_bis.pt").numpy()
        noise_pred = np.load(f"/media/mp274748/Transcend/data/LSC/RUN9/noise_file_RUN9/pulses_pred_d0_{channel:03}.npy")
        data_noise = BinaryDatasetDenoised(find_file(f"000813_20230628T161508_{channel:03}_???.bin"),noise_pred[:,noise_pred.shape[1]//2-win_length // 2 -shift:noise_pred.shape[1]//2+win_length // 2 -shift]*Global_stds[channels.index(channel)], win_length, positions_single, win_shift=win_length // 2 - shift)
    else :
        dataset_single_name = "pup_n1-d0_000813_20230628T161508"
        file_path_single, path_pos_single = ds.find_files(dataset_single_name, channel,pos_prefix="_bis", file_suffix= "", differenciate=False,
                         specific_subdir="RUN9_pulse_injected_new/")
        pos_file_single = np.loadtxt(path_pos_single)[:15000]
        positions_single = (pos_file_single[:, 0] // 4).astype(int)
        data_noise = BinaryDataset(find_file(f"000813_20230628T161508_{channel:03}_???.bin"), win_length, positions_single, win_shift=win_length // 2 - shift)
    return data_noise