from statistics import multimode
import warnings

import numpy as np
import pandas as pd
from impedance.models.circuits import CustomCircuit

from electrical_test_analyzer.dataframe_operations import naive_df_area_normalization, df_to_common_row
from electrical_test_analyzer.test_analyzers.eis_fit_functions import rcpe_circuit_string, get_rough_arc_indices, create_rcpe_guess_array, fit_rcpe_with_timeout, rcpe_fit_result_columns
from electrical_test_analyzer.test_analyzers.test_analyzer_template import TestAnalyzer




class EISFitter(TestAnalyzer):



    @staticmethod
    def get_rough_num_arcs_df(eis_df, min_index_separation):

        name_scan_arcs_filepath = []
        for (sample_name, scan_count, filepath), scan_df in eis_df.groupby(['Sample name', 'Scan count', 'filepath']):
            indices = get_rough_arc_indices(*scan_df[['Re Z (Ohm)', '-Im Z (Ohm)']].to_numpy().T, min_index_separation)
            name_scan_arcs_filepath.append((sample_name, scan_count, indices.size - 1, filepath))

        rough_num_arcs_df = pd.DataFrame(name_scan_arcs_filepath, columns=['Sample name', 'Scan count', 'Rough num arcs', 'filepath'])
        
        return rough_num_arcs_df



    @classmethod
    def split_df(cls, input_df, **kwargs):
        return [single_file_eis_df for _, single_file_eis_df in input_df.groupby(['Sample name', 'filepath'])] # Need all scan counts in single file to 
    


    @classmethod
    def analyze_subset_df(cls, single_file_eis_df, min_index_separation=7, num_arcs=None, num_processes=3, timeout=10, tqdm_kwargs=None, **kwargs):

        common_row = df_to_common_row(single_file_eis_df)

        scan_counts = []
        start_times = []
        start_capacities = []

        min_freqs = []
        max_freqs = []

        freq_list = []
        re_z_list = []
        minus_im_z_list = []

        for scan_count, single_scan_df in single_file_eis_df.groupby('Scan count'):
            scan_counts.append(scan_count)
            start_times.append(single_scan_df['Time (s)'].iloc[0] / 3600)
            start_capacities.append(single_scan_df['Capacity (mAh)'].iloc[0])
            
            freq = single_scan_df['Frequency (Hz)'].to_numpy()
            min_freqs.append(np.min(freq))
            max_freqs.append(np.max(freq))

            freq_list.append(freq)
            re_z_list.append(single_scan_df['Re Z (Ohm)'].to_numpy())
            minus_im_z_list.append(single_scan_df['-Im Z (Ohm)'].to_numpy())

        zipped_vals = list(zip(scan_counts, start_times, start_capacities, freq_list, re_z_list, minus_im_z_list))
        zipped_vals.sort(key=lambda x : x[0])
        scan_counts, start_times, start_capacities, freq_list, re_z_list, minus_im_z_list = zip(*zipped_vals)

        indices_list = [get_rough_arc_indices(re_z, minus_im_z, min_index_separation) for re_z, minus_im_z in zip(re_z_list, minus_im_z_list)]

        if num_arcs is None:
            num_arcs = multimode([indices.size - 1 for indices in indices_list])[0]
        
        circuit_string = rcpe_circuit_string(num_arcs)
        guess_array = create_rcpe_guess_array(freq_list, re_z_list, indices_list, num_arcs)

        fit_results = fit_rcpe_with_timeout(freq_list, re_z_list, minus_im_z_list, circuit_string, guess_array, num_processes, timeout, tqdm_kwargs)
        scan_params = np.stack((scan_counts, start_times, start_capacities, np.full(np.asarray(scan_counts).shape, num_arcs), min_freqs, max_freqs), axis=1)
        fit_results = np.concatenate((scan_params, fit_results), axis=1)

        fit_results_df = pd.DataFrame(fit_results, columns=['Scan count', 'Scan start time (h)', 'Scan start capacity (mAh)', 'Num fitted arcs', 'Min freq (Hz)', 'Max freq (Hz)'] + rcpe_fit_result_columns(num_arcs))

        repeated_common_row_df = pd.DataFrame(len(fit_results_df) * [common_row]).reset_index(drop=True) # Index must be reset and drop=True for concatenation to work
        output_subset_df = pd.concat((repeated_common_row_df, fit_results_df), axis=1)

        # output_subset_df = naive_df_area_normalization(output_subset_df)

        return output_subset_df
    

    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        output_df = naive_df_area_normalization(output_df)
        return output_df




class EISFitTracer(TestAnalyzer):

    @classmethod
    def split_df(cls, input_df, **kwargs):
        return [row for _, row in input_df.iterrows()]


    @classmethod
    def analyze_subset_df(cls, row, num_freqs=100, **kwargs):
        

        num_fitted_arcs = int(row['Num fitted arcs'])
        circuit_params = row[rcpe_fit_result_columns(num_fitted_arcs)].to_numpy(dtype=float).flatten()[:-1]
        
        if np.any(np.isnan(circuit_params)):
            return None
        
        circuit_string = rcpe_circuit_string(num_fitted_arcs)
        freq_fit = np.logspace(np.log10(row['Min freq (Hz)']), np.log10(row['Max freq (Hz)']), num_freqs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            z_fit = np.array([CustomCircuit(circuit_string, initial_guess=circuit_params).predict(freq_fit)]).squeeze()
        
        re_z_fit = np.real(z_fit)
        minus_im_z_fit = -np.imag(z_fit)

        results_subset_df = pd.DataFrame(np.stack((freq_fit, re_z_fit, minus_im_z_fit), axis=1), columns=['Frequencies (Hz)', 'Re Z Fit (Ohm)', '-Im Z Fit (Ohm)'])
        repeated_row_df = pd.DataFrame(len(results_subset_df) * [row]).reset_index(drop=True) # Index must be reset and drop=True for concatenation to work
        output_subset_df = pd.concat((repeated_row_df, results_subset_df), axis=1)
        # output_subset_df = naive_df_area_normalization(output_subset_df)
        
        return output_subset_df
    

    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        output_df = naive_df_area_normalization(output_df)
        return output_df