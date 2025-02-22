import os

import numpy as np
import pandas as pd

from electrical_test_analyzer.dataframe_operations import concatenate_dfs_with_continuing_columns, df_to_column_value_sequences, naive_df_area_normalization, pd_filetype_agnostic_read
from electrical_test_analyzer.file_interfaces import file_interface_extension_map
from electrical_test_analyzer.test_analyzers.test_analyzer_template import TestAnalyzer, FileAnalyzer



def add_inactive_areal_density(data_df, component_properties_df):


    component_thickness_column_pairs = (
        ('Cathode type', 'Cathode thickness (mm)'),
        ('Separator type', 'Separator thickness (mm)'),
        ('Anode type', 'Anode thickness (mm)')
    )

    component_areal_densities = []
    for component_column, thickness_column in component_thickness_column_pairs:
        densities = data_df[component_column].map(component_properties_df.set_index('Component name')['Density (g/cm3)'])
        thicknesses = data_df[thickness_column]
        component_areal_density = densities * thicknesses * 0.1 # g/cm2
        component_areal_densities.append(component_areal_density)
    incative_component_areal_density = sum(component_areal_densities)
    data_df['Inactive component areal density (g/cm2)'] = incative_component_areal_density

    return data_df





class MetricAnalyzer(FileAnalyzer):

    time_cols = ['Total time (h)', 'Polarize only total time (h)']
    cap_cols = ['Total capacity (mAh)', 'Polarize only total capacity (mAh)']
    energy_cols = ['Total energy (Wh)', 'Polarize only total energy (Wh)']

    headers = [item for tup in zip(time_cols, cap_cols, energy_cols) for item in tup]


    @classmethod
    def filepath_to_df(cls, filepath, interface=None, **kwargs):
        _, ext = os.path.splitext(filepath)
        interface = interface or file_interface_extension_map[ext]
        raw_data_df = interface.filepath_to_standard_data_df(filepath, usecols=['Time (s)', 'Capacity (mAh)', 'Energy (Wh)', 'Current (mA)', 'Voltage (V)', 'Frequency (Hz)'], **kwargs)
        polarize_only_df = concatenate_dfs_with_continuing_columns([polarize_step_container[0].copy() for polarize_step_container in df_to_column_value_sequences(raw_data_df, 'MODE', ['POLARIZE'])], continuing_columns=['Time (s)', 'Capacity (mAh)', 'Energy (Wh)'], ignore_index=True)

        summary_line = np.array([np.max(df[key]) - np.min(df[key]) for df in [raw_data_df, polarize_only_df] for key in ['Time (s)', 'Capacity (mAh)', 'Energy (Wh)']])

        # Convert time to hours
        summary_line[0] /= 3600
        summary_line[3] /= 3600
        
        results_df = pd.DataFrame([summary_line], columns=cls.headers)
        
        return results_df



    @classmethod
    def analyze(cls, input_df, specific_capacity, component_properties_manifest_filepath, read_kwargs=None, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        output_df = naive_df_area_normalization(output_df)

        read_kwargs = read_kwargs or {}

        
        component_properties_df = pd_filetype_agnostic_read(component_properties_manifest_filepath, **read_kwargs)
        output_df = add_inactive_areal_density(output_df, component_properties_df)


        # TODO: put in loop

        # Total
        active_areal_density = output_df['Total capacity (mAh/cm2)'] / specific_capacity
        system_areal_density = active_areal_density + output_df['Inactive component areal density (g/cm2)']

        energy_density_active_basis = 1000 * output_df['Total energy (Wh/cm2)'] / active_areal_density
        energy_density_system_basis = 1000 * output_df['Total energy (Wh/cm2)'] / system_areal_density
        
        power_density_active_basis = energy_density_active_basis / output_df['Total time (h)']
        power_density_system_basis = energy_density_system_basis / output_df['Total time (h)']

        output_df['Energy density active basis (Wh/kg)'] = energy_density_active_basis
        output_df['Energy density system basis (Wh/kg)'] = energy_density_system_basis

        output_df['Average power density active basis (W/kg)'] = power_density_active_basis
        output_df['Average power density system basis (W/kg)'] = power_density_system_basis


        # Polarize only
        active_areal_density = output_df['Polarize only total capacity (mAh/cm2)'] / specific_capacity
        system_areal_density = active_areal_density + output_df['Inactive component areal density (g/cm2)']

        energy_density_active_basis = 1000 * output_df['Polarize only total energy (Wh/cm2)'] / active_areal_density
        energy_density_system_basis = 1000 * output_df['Polarize only total energy (Wh/cm2)'] / system_areal_density
        
        power_density_active_basis = energy_density_active_basis / output_df['Polarize only total time (h)']
        power_density_system_basis = energy_density_system_basis / output_df['Polarize only total time (h)']

        output_df['Polarize only energy density active basis (Wh/kg)'] = energy_density_active_basis
        output_df['Polarize only energy density system basis (Wh/kg)'] = energy_density_system_basis

        output_df['Polarize only average power density active basis (W/kg)'] = power_density_active_basis
        output_df['Polarize only average power density system basis (W/kg)'] = power_density_system_basis

        return output_df



















##################################################################################################        
# Work in progress
##################################################################################################
class EnergyPowerAnalyzer(TestAnalyzer):
    

    headers = ['Total time (h)', 'Total capacity (mAh)', 'Total energy (Wh)']

    @classmethod
    def split_df(cls, input_df, component_properties_df=None, **kwargs):
        input_df = add_inactive_areal_density(input_df, component_properties_df)
        return [row for _, row in input_df.iterrows()]
    
    @classmethod
    def analyze_subset_df(cls, row, required_energy_density=None, polarize_only=False, **kwargs):

        filepath = row['filepath']
        active_area = row['Active area (cm2)']

        _, ext = os.path.splitext(filepath)
        interface = interface or file_interface_extension_map[ext]
        raw_data_df = interface.filepath_to_standard_data_df(filepath, usecols=['Time (s)', 'Capacity (mAh)', 'Energy (Wh)', 'Current (mA)', 'Voltage (V)'], **kwargs)
        if polarize_only:
            raw_data_df = concatenate_dfs_with_continuing_columns([polarize_step_container[0].copy() for polarize_step_container in df_to_column_value_sequences(raw_data_df, 'MODE', ['POLARIZE'])], continuing_columns=['Time (s)', 'Capacity (mAh)', 'Energy (Wh)'], ignore_index=True)

        time, capacity, energy, current, voltage = raw_data_df[['Time (s)', 'Capacity (mAh)', 'Energy (Wh)', 'Current (mA)', 'Voltage (V)']].to_numpy().T
        power = voltage * current

        #######################################

        summary_line = np.array([np.max(raw_data_df[key]) - np.min(raw_data_df[key]) for key in ['Time (s)', 'Capacity (mAh)', 'Energy (Wh)']])
        summary_line[0] /= 3600

        results_df = pd.DataFrame([summary_line], columns=cls.headers)

        return results_df
        # issue with longest horizontal line segment idea is that it doesn't account for mass of sodium
        # try binary search on sorted y values lol

    
    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        output_df = naive_df_area_normalization(output_df)
        return output_df





