import multiprocessing as mp

import numpy as np
from scipy import signal
from tqdm import tqdm

from impedance.models.circuits import CustomCircuit




def rcpe_circuit_string(num_arcs):
    return '-'.join(['R0'] + [f'p(R{i},CPE{i})' for i in range(1, num_arcs + 1)])

def rcpe_fit_result_columns(num_arcs):
    columns = ['R0 (Ohm)'] + [item for triple in [(f'R{i} (Ohm)', f'C{i} (F)', f'Alpha{i}') for i in range(1, num_arcs + 1)] for item in triple] + ['MSE']
    return columns



def get_rough_arc_indices(re_z, minus_im_z, min_index_separation):
    # Assumes points are ordered from high to low frequency

    idx = [0]

    for _ in range(re_z.size):

        modified_re_z = re_z - re_z[idx[-1]]
        modified_minus_im_z = minus_im_z - minus_im_z[idx[-1]]
        modified_phase = np.arctan2(modified_minus_im_z, modified_re_z)
        peak_idx, _ = signal.find_peaks(-modified_phase)

        if not np.any(peak_idx > idx[-1] + min_index_separation):
            break
        
        idx.append(np.min(peak_idx[peak_idx > idx[-1] + min_index_separation]))
    
    if re_z.size > 1:
        idx.append(re_z.size - 1)
    
    return np.array(idx)



def guess_rcpe_params(freq, re_z, indices):

    rs = np.diff(re_z[indices])
    fs = freq[indices[:-1]] + 0.5 * np.diff(freq[indices])
    cs = 1 / (rs * fs)

    guess = np.array([re_z[indices[0]]] + sum([[r, c, 1.0] for r, c in zip(rs, cs)], start=[]))
    return guess





def create_rcpe_guess_array(freq_list, re_z_list, indices_list, num_fit_arcs):

    # Check if there is at least one scan that matches num_fit_arcs
    num_arcs = np.array([indices.size - 1 for indices in indices_list])
    if not np.any(num_arcs == num_fit_arcs):
        raise ValueError('No indices match num_fit_arcs.')
    
    # Mask scans which match num_fit_arcs and find indices
    native_num_arcs_mask = num_arcs == num_fit_arcs
    native_indices = np.argwhere(native_num_arcs_mask).squeeze()

    # Initialize guess array
    guess_array = np.ones((len(re_z_list), 1 + 3 * num_fit_arcs))

    # Populate guess array with scans matching num_fit_arcs
    guess_array[native_num_arcs_mask] = np.stack([guess_rcpe_params(freq, re_z, indices) for re_z, freq, indices in zip(re_z_list, freq_list, indices_list) if indices.size - 1 == num_fit_arcs])

    # Populate guess array for other scans with nearest matching scan data
    guess_array[~native_num_arcs_mask] = [guess_array[native_indices[np.argmin((native_indices - i) ** 2)]] for i, b in enumerate(native_num_arcs_mask) if not b]

    return guess_array



def circuit_fit_process(args):

    freq, z, circuit = args

    try:
        circuit.fit(freq, z)
        params = circuit.parameters_

        z_predict = circuit.predict(freq)
        mse = np.mean(np.abs(z - z_predict) ** 2)
        # mse = np.sum(np.abs(z - z_predict)) / z.size

        return np.concatenate((params, np.array([mse])))
    
    except:
        return np.full(len(circuit.initial_guess) + 1, np.nan)



def fit_rcpe_with_timeout(freq_list, re_z_list, minus_im_z_list, circuit_string, guess_array, num_processes=3, timeout=10, tqdm_kwargs=None):
    num_processes = (num_processes - 1) % mp.cpu_count() + 1

    tqdm_kwargs = tqdm_kwargs or {}

    z_list = [re_z - 1j * minus_im_z for re_z, minus_im_z in zip(re_z_list, minus_im_z_list)]
    circuit_list = [CustomCircuit(circuit_string, initial_guess=guess) for guess in guess_array]
    args = list(zip(freq_list, z_list, circuit_list))

    with mp.Pool(processes=num_processes) as pool:
        async_results = [pool.apply_async(circuit_fit_process, (arg,)) for arg in args]
    
        results = []
        for result in tqdm(async_results, **tqdm_kwargs):
            try:
                results.append(result.get(timeout=timeout))
            except Exception as e:
                results.append(np.full(guess_array.shape[1] + 1, np.nan))
    
    results = np.stack(results)

    return results