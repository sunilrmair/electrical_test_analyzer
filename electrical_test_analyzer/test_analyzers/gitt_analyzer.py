import os

import numpy as np
import pandas as pd

from electrical_test_analyzer.dataframe_operations import df_to_column_value_sequences, naive_df_area_normalization
from electrical_test_analyzer.file_interfaces import file_interface_extension_map
from electrical_test_analyzer.test_analyzers.test_analyzer_template import FileAnalyzer



class GITTAnalyzer(FileAnalyzer):



    headers = [

        'Pulse count',

        'Polarize start time (h)',
        'Polarize end time (h)',

        'Rest start time (h)',
        'Rest end time (h)',

        'Polarize start capacity (mAh)',
        'Polarize end capacity (mAh)',

        'Mean polarization current (mA)',
        'Standard deviation polarization current (mA)',
        'Median polarization current (mA)',
        'Mean last 10 polarization current points (mA)',

        'Mean polarization voltage (V)',
        'Standard deviation polarization voltage (V)',
        'Median polarization voltage (V)',
        'Mean last 10 polarization voltage points (V)',

        'Mean rest voltage (V)',
        'Standard deviation rest voltage (V)',
        'Median rest voltage (V)',
        'Mean last 10 rest voltage points (V)',

        'Mean last 10 DC resistance (Ohm)'

    ]

    

    @staticmethod
    def summarize_values(values, last_n):

        mean_value = np.mean(values)
        std_value = np.std(values)
        median_value = np.median(values)
        if values.size <= last_n:
            mean_last_n = mean_value
        else:
            mean_last_n = np.mean(values[-last_n:])

        return np.array([mean_value, std_value, median_value, mean_last_n])



    @classmethod
    def polarize_rest_step_to_summary(cls, polarize_df, rest_df):

        polarize_start_time = polarize_df['Time (s)'].iloc[0] / 3600
        polarize_end_time =  polarize_df['Time (s)'].iloc[-1] / 3600
        rest_start_time = rest_df['Time (s)'].iloc[0] / 3600
        rest_end_time = rest_df['Time (s)'].iloc[-1] / 3600

        polarize_start_capacity = polarize_df['Capacity (mAh)'].iloc[0]
        polarize_end_capacity = polarize_df['Capacity (mAh)'].iloc[-1]

        polarize_currents = polarize_df['Current (mA)'].to_numpy()
        mean_polarize_current, std_polarize_current, median_polarize_current, mean_last_10_polarize_current = cls.summarize_values(polarize_currents, last_n=10)

        polarize_voltages = polarize_df['Voltage (V)'].to_numpy()
        mean_polarize_voltage, std_polarize_voltage, median_polarize_voltage, mean_last_10_polarize_voltage = cls.summarize_values(polarize_voltages, last_n=10)

        rest_voltages = rest_df['Voltage (V)'].to_numpy()

        mean_rest_voltage, std_rest_voltage, median_rest_voltage, mean_last_10_rest_voltage = cls.summarize_values(rest_voltages, last_n=10)

        mean_last_10_dc_resistance = np.abs((mean_last_10_polarize_voltage - mean_last_10_rest_voltage) / (0.001 * mean_last_10_polarize_current))



        summary_line = np.array([

            polarize_start_time,
            polarize_end_time,

            rest_start_time,
            rest_end_time,

            polarize_start_capacity,
            polarize_end_capacity,

            mean_polarize_current,
            std_polarize_current,
            median_polarize_current,
            mean_last_10_polarize_current,

            mean_polarize_voltage,
            std_polarize_voltage,
            median_polarize_voltage,
            mean_last_10_polarize_voltage,

            mean_rest_voltage,
            std_rest_voltage,
            median_rest_voltage,
            mean_last_10_rest_voltage,

            mean_last_10_dc_resistance

        ])

        return summary_line
    


    @classmethod
    def filepath_to_df(cls, filepath, interface=None, **kwargs):
        _, ext = os.path.splitext(filepath)
        interface = interface or file_interface_extension_map[ext]
        raw_data_df = interface.filepath_to_standard_data_df(filepath, usecols=['Time (s)', 'Capacity (mAh)', 'Current (mA)', 'Voltage (V)'], **kwargs)
        
        polarize_rest_steps = df_to_column_value_sequences(raw_data_df, 'MODE', ['POLARIZE', 'REST'])

        results_content = np.stack([np.concatenate((np.array([pulse_count]), cls.polarize_rest_step_to_summary(polarize_step, rest_step))) for pulse_count, (polarize_step, rest_step) in enumerate(polarize_rest_steps)])
        results_df = pd.DataFrame(results_content, columns=cls.headers)

        return results_df



    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        output_df = naive_df_area_normalization(output_df)
        return output_df