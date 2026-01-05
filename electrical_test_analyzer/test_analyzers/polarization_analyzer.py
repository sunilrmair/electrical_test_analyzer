
import numpy as np
import pandas as pd

from electrical_test_analyzer.dataframe_operations import df_to_common_row
from electrical_test_analyzer.test_analyzers.test_analyzer_template import TestAnalyzer, FileAnalyzer


# Depends on 'Ns' column which may be unique to mpt format

class PolarizationAnalyzer(TestAnalyzer):

    @classmethod
    def split_df(cls, input_df, groupby_kwargs=None, **kwargs):
        groupby_kwargs = groupby_kwargs or dict(by='filepath')
        return [subset_df for _, subset_df in input_df.groupby(**groupby_kwargs)]


    @classmethod
    def analyze_subset_df(cls, subset_df, **kwargs):
        
        common_row_dict = df_to_common_row(subset_df).to_dict()
        subset_df['Cycle'] = (subset_df['Ns'].diff() < 0).cumsum()
        
        row_list = []
        for cycle_num, single_polarization_df in subset_df.groupby(by='Cycle'):
            x = single_polarization_df['Current (mA/cm2)'].to_numpy()
            y = single_polarization_df['Voltage (V)'].to_numpy()
            p = np.polyfit(x, y, deg=1)

            row_specific_data = {
                'Cycle' : cycle_num,
                'OCV (V)' : p[1],
                'ASR (Ohm-cm2)' : 1000 * p[0]
            } 
            row_list.append(common_row_dict | row_specific_data)
        
        return pd.DataFrame(row_list)


    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        return output_df