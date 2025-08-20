import os

import numpy as np
import pandas as pd

from electrical_test_analyzer.dataframe_operations import df_to_column_value_sequences, naive_df_area_normalization
from electrical_test_analyzer.file_interfaces import file_interface_extension_map
from electrical_test_analyzer.test_analyzers.test_analyzer_template import FileAnalyzer




def clip_to_first_quadrant(re_z, minus_im_z, *args):
    mask = np.logical_and(re_z >= 0, minus_im_z >= 0)
    masked_args = [arg[mask] for arg in args]
    return re_z[mask], minus_im_z[mask], *masked_args




class EISBasicAnalyzer(FileAnalyzer):

    headers = ['Scan count', 'Scan start time (h)', 'Scan start capacity (mAh)', 'Minimum Re Z in first quadrant (Ohm)']

    @classmethod
    def eis_step_to_row(cls, scan_count, eis_df, **kwargs):

        t, c, freq, re_z, minus_im_z = eis_df[['Time (s)', 'Capacity (mAh)', 'Frequency (Hz)', 'Re Z (Ohm)', '-Im Z (Ohm)']].to_numpy().T

        start_time = t[0] / 3600
        start_cap = c[0]
        re_z, minus_im_z = clip_to_first_quadrant(re_z, minus_im_z)
        r0 = np.min(re_z)

        return np.array([scan_count, start_time, start_cap, r0])



    @classmethod
    def filepath_to_df(cls, filepath, interface=None, **kwargs):

        _, ext = os.path.splitext(filepath)
        interface = interface or file_interface_extension_map[ext]
        raw_data_df = interface.filepath_to_standard_data_df(filepath, **kwargs)
        
        nested_eis_steps = df_to_column_value_sequences(raw_data_df, 'MODE', ['EIS'])
        eis_steps = [x[0] for x in nested_eis_steps]

        results_row = np.stack([cls.eis_step_to_row(scan_count, eis_df, **kwargs) for scan_count, eis_df in enumerate(eis_steps)])
        results_df = pd.DataFrame(results_row, columns=cls.headers)

        return results_df
    

    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        output_df = naive_df_area_normalization(output_df)
        return output_df






class EISLoader(FileAnalyzer):


    headers = ['Scan count', 'Time (s)', 'Capacity (mAh)', 'Frequency (Hz)', 'Re Z (Ohm)', '-Im Z (Ohm)']
    
    
    
    @classmethod
    def eis_step_to_table(cls, scan_count, eis_df, pretreat_func=None, **kwargs):
        pretreat_func = pretreat_func or clip_to_first_quadrant

        if 'Capacity (mAh)' not in eis_df.columns:
            eis_df['Capacity (mAh)'] = np.nan

        t, c, freq, re_z, minus_im_z = eis_df[['Time (s)', 'Capacity (mAh)', 'Frequency (Hz)', 'Re Z (Ohm)', '-Im Z (Ohm)']].to_numpy().T
        re_z, minus_im_z, freq, t, c = pretreat_func(re_z, minus_im_z, freq, t, c)
        summary_table = np.stack((np.full(freq.shape, scan_count), t, c, freq, re_z, minus_im_z), axis=1)
        
        return summary_table

    @classmethod
    def filepath_to_df(cls, filepath, interface=None, **kwargs):
        _, ext = os.path.splitext(filepath)
        interface = interface or file_interface_extension_map[ext]
        raw_data_df = interface.filepath_to_standard_data_df(filepath, **kwargs)
        
        nested_eis_steps = df_to_column_value_sequences(raw_data_df, 'MODE', ['EIS'])
        eis_steps = []
        for x in nested_eis_steps:
            if 'cycle number' in x[0].columns:
                eis_steps.extend([df for _, df in x[0].groupby('cycle number')])
            else:
                eis_steps.append(x[0])
        # eis_steps = [x[0] for x in nested_eis_steps]

        results_content = np.concatenate([cls.eis_step_to_table(scan_count, eis_df, **kwargs) for scan_count, eis_df in enumerate(eis_steps)])
        results_df = pd.DataFrame(results_content, columns=cls.headers)

        return results_df



    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        output_df = naive_df_area_normalization(output_df)
        return output_df



