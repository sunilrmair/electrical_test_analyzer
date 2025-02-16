import os
import warnings

import pandas as pd

from electrical_test_analyzer.dataframe_operations import pd_filetype_agnostic_read
from electrical_test_analyzer.helper_functions import ensure_iterable





class QueryManager:
    """Manages query operations for experimental data files, including filtering and searching."""


    valid_filetypes = [
        '.mpt'
    ]



    def __init__(self, cell_parameters_db_filepath, test_parameters_db_filepath):
        """Initializes QueryManager by loading and merging test and cell parameters into `parameters_df`.

        Args:
            cell_parameters_db_filepath (str | list[str]): File path(s) to the cell parameters database.
            test_parameters_db_filepath (str | list[str]): File path(s) to the test parameters database.
        """

        cell_parameters_df = pd.concat([pd_filetype_agnostic_read(filepath) for filepath in ensure_iterable(cell_parameters_db_filepath)])
        test_parameters_df = pd.concat([pd_filetype_agnostic_read(filepath) for filepath in ensure_iterable(test_parameters_db_filepath)])

        self.parameters_df = pd.merge(test_parameters_df, cell_parameters_df, on='Sample name', how='left')

        self.filtersets = []



    @property
    def num_tests(self):
        """Returns the number of tests in the dataset.

        Returns:
            int: Number of tests in `parameters_df`.
        """
        return len(self.parameters_df)
    


    def add_filterset(self, filterset):
        """Adds one or more filtersets to be applied later.

        Args:
            filterset (FilterSet | list[FilterSet]): A single filterset or a list of filtersets to add.
        """
        self.filtersets.extend(ensure_iterable(filterset))



    def apply_filtersets(self):
        """Applies all added filtersets to filter `parameters_df`. If no filtersets are present, a warning is issued."""

        if self.filtersets:
            self.parameters_df = pd.concat([filterset.apply(self.parameters_df) for filterset in self.filtersets]).drop_duplicates()
            self.parameters_df.reset_index(inplace=True)
            self.filtersets = []
        else:
            warnings.warn('No filtersets were applied.')


    
    def search(self, root_directory):
        """Searches for files matching dataset entries within the specified root directory.

        Args:
            root_directory (str | Pathlike | list[str | Pathlike]): Root directory or directories to search in.

        Returns:
            pandas.DataFrame: A copy of `parameters_df` with an additional 'filepath' column.
        """

        results_df = self.parameters_df.copy(deep=True)
        results_df['filepath'] = ''

        for root in ensure_iterable(root_directory):
            for dirpath, _, files in os.walk(root):
                for file in files:
                    filename, fileext = os.path.splitext(file)
                    if fileext in self.valid_filetypes:
                    
                        sample_name_in_filename_mask = results_df['Sample name'].apply(lambda x: x in filename)
                        file_specifier_in_filename_mask = results_df['File specifier'].apply(lambda x: x in filename)
                        combined_mask = sample_name_in_filename_mask & file_specifier_in_filename_mask

                        if combined_mask.sum() > 1:
                            warnings.warn(f'The following file matched multiple sample names and file specifiers: {file}')

                        results_df.loc[combined_mask, 'filepath'] = os.path.join(dirpath, file)
        
        empty_filepath_mask = results_df['filepath'] == ''
        if any(empty_filepath_mask):
            popped_rows = results_df[empty_filepath_mask]
            results_df = results_df[~empty_filepath_mask]
            dropped_row_sample_names_and_file_specifiers = tuple(tuple(row) for row in popped_rows[['Sample name', 'File specifier']].values)
            warnings.warn(f'The following sample names and file specifiers did not match any files: {dropped_row_sample_names_and_file_specifiers}')

        return results_df