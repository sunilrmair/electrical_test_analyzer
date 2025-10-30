import glob
import os

import pandas as pd

from electrical_test_analyzer.dataframe_operations import concatenate_dfs_with_continuing_columns, naive_df_area_normalization
from electrical_test_analyzer.test_analyzers.test_analyzer_template import TestAnalyzer
from electrical_test_analyzer.test_analyzers.raw_data_loader import RawDataLoader



class LifetimeAnalyzer(TestAnalyzer):


    
    @classmethod
    def split_df(cls, input_df, **kwargs):
        return [row for _, row in input_df.iterrows()]


    @classmethod
    def analyze_subset_df(cls, row, data_directory=None, filetypes=['mpt'], sort_key=os.path.getmtime, continuing_columns=None, interval=None, **kwargs):
        
        sample_name = row['Sample name']

        # search data directory
        filepaths = []
        for filetype in filetypes:
            filepaths += glob.glob(data_directory + f'/**/*{sample_name}*.{filetype}', recursive=True)
        filepaths = sorted(filepaths, key=sort_key)

        results_subset_df = concatenate_dfs_with_continuing_columns([RawDataLoader.filepath_to_df(filepath) for filepath in filepaths], continuing_columns=continuing_columns, interval=interval)
        repeated_row_df = pd.DataFrame(len(results_subset_df) * [row]).reset_index(drop=True) # Index must be reset and drop=True for concatenation to work
        output_subset_df = pd.concat((repeated_row_df, results_subset_df), axis=1)

        return output_subset_df
    

    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        output_df = naive_df_area_normalization(output_df)
        return output_df