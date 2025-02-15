# import re

import numpy as np
import pandas as pd


from electrical_test_analyzer.helper_functions import get_title_and_units





def pd_filetype_agnostic_read(filepath, **kwargs):

    ext = filepath.split('.')[-1].lower()
    
    if ext == 'csv':
        return pd.read_csv(filepath, **kwargs)
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(filepath, **kwargs)
    elif ext == 'json':
        return pd.read_json(filepath, **kwargs)
    elif ext == 'parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif ext == 'feather':
        return pd.read_feather(filepath, **kwargs)
    elif ext == 'h5':
        return pd.read_hdf(filepath, **kwargs)
    elif ext == 'html':
        return pd.read_html(filepath, **kwargs)[0]  # Returns a list; take the first table
    elif ext == 'xml':
        return pd.read_xml(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: .{ext}")





def df_to_column_value_sequences(df, column, sequence):

    sequence_length = len(sequence)
    sequence = np.asarray(sequence)

    dfs_grouped_by_consecutive_column_values = df.groupby((df[column] != df[column].shift(1)).cumsum() - 1)
    column_group_values = np.array([group_df[column].iloc[0] for _, group_df in dfs_grouped_by_consecutive_column_values])

    consecutive_group_values = np.stack([column_group_values[i:-sequence_length+i] for i in range(sequence_length)], axis=-1)
    match_mask = np.pad(np.all(consecutive_group_values == sequence, axis=-1), (0, sequence_length), constant_values=False)

    matched_df_sequences = [[dfs_grouped_by_consecutive_column_values.get_group(j) for j in range(i, i + sequence_length)] for i in range(match_mask.size) if match_mask[i]]
    return matched_df_sequences

    



def concatenate_dfs_with_continuing_columns(dfs, continuing_columns=None, **kwargs):

    continuing_columns = continuing_columns or []
    default_kwargs = dict(ignore_index=True)
    kwargs = {**default_kwargs, **kwargs}

    # Translate each continuing column to start at 0
    for df in dfs:
        for column in continuing_columns:
            if column in df.columns:
                df[column] = df[column] - df[column].iloc[0]
    

    # Calculate cumulative value per df
    cumulative_column_values = {column : np.zeros(len(dfs)) for column in continuing_columns}
    for i, df in enumerate(dfs[:-1]):
        for column in continuing_columns:
            if column in df.columns:
                cumulative_column_values[column][i + 1] = cumulative_column_values[column][i] + df[column].iloc[-1]
            else:
                cumulative_column_values[column][i + 1] = cumulative_column_values[column][i]
    

    # Add cumulative values
    for i, df in enumerate(dfs):
        for column in continuing_columns:
            if column in df.columns:
                df[column] += cumulative_column_values[column][i]
    

    # Concatenate
    concatenated_dfs = pd.concat(dfs, **kwargs)

    return concatenated_dfs
    




def naive_df_area_normalization(df, active_area='Active area (cm2)', **kwargs):
        """Processes DataFrame columns by scaling specific columns based on an active area.

        Args:
            df (pandas.DataFrame): The DataFrame with columns to scale
            active_area_key (str | float | int, optional): The column name representing the active area in cm2 or a number representing the active area in cm2. Defaults to 'Active area (cm2)'.

        Returns:
            pandas.DataFrame : The DataFrame with additional scaled columns.
        """


        if isinstance(active_area, str):
            active_area = df[active_area]


        divide_by_area_units = [
            'mAh',
            'mA',
            'mW',
            'mWh',
            'Wh',
            'F'
        ]

        multiply_by_area_units = [
            'Ohm'
        ]

        for column in df.columns:
            
            if any('(' + substring + ')' in column for substring in divide_by_area_units):
                title, units = get_title_and_units(column)
                if title + ' (' + units + '/cm2)' not in df.columns:
                    df[title + ' (' + units + '/cm2)'] = df[column] / active_area
            
            if any('(' + substring + ')' in column for substring in multiply_by_area_units):
                title, units = get_title_and_units(column)
                if title + ' (' + units + '-cm2)' not in df.columns:
                    df[title + ' (' + units + '-cm2)'] = df[column] * active_area
            
        return df





def df_to_common_row(df):
    same_value_columns = df.columns[df.nunique() == 1]
    return df[same_value_columns].iloc[0]






class FilterSet:


    def __init__(self, query_filters=None, isin_filters=None, not_in_filters=None, between_filters=None, lambda_filters=None, null_check_filters=None):

        self.query_filters = query_filters or []
        self.isin_filters = isin_filters or {}
        self.not_in_filters = not_in_filters or {}
        self.between_filters = between_filters or {}
        self.lambda_filters = lambda_filters or []
        self.null_check_filters = null_check_filters or {}


    def add_query(self, query_str):
        self.query_filters.append(query_str)


    def add_isin(self, column, values):
        self.isin_filters[column] = values


    def add_not_in(self, column, values):
        self.not_in_filters[column] = values


    def add_between(self, column, low, high):
        self.between_filters[column] = (low, high)


    def add_lambda(self, condition_func):
        self.lambda_filters.append(condition_func)


    def add_null_check(self, column, is_null=False):
        self.null_check_filters[column] = is_null


    def apply(self, df):

        # Apply query filters
        for query_str in self.query_filters:
            df = df.query(query_str)

        # Apply isin filters
        for col, values in self.isin_filters.items():
            df = df[df[col].isin(values)]

        # Apply not_in filters
        for col, values in self.not_in_filters.items():
            df = df[~df[col].isin(values)]

        # Apply between filters
        for col, (low, high) in self.between_filters.items():
            df = df[df[col].between(low, high)]

        # Apply lambda filters
        for func in self.lambda_filters:
            df = df[func(df)]

        # Apply null check filters
        for col, is_null in self.null_check_filters.items():
            if is_null:
                df = df[df[col].isna()]
            else:
                df = df[df[col].notna()]

        return df