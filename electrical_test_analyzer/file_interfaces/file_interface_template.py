from abc import ABC, abstractmethod

import pandas as pd





class FileInterface(ABC):
    """A template for a collection of methods for extracting and manipulating data from files."""



    @classmethod
    @abstractmethod
    def filepath_to_raw_data_df(cls, filepath, **kwargs):
        """Loads the raw data of the file into a DataFrame.

        Args:
            filepath (str | Pathlike): The file to read.

        Returns:
            pandas.DataFrame: A DataFrame containing the file's raw data.
        """
        pass



    @classmethod
    @abstractmethod
    def filepath_to_standard_data_df(cls, filepath, **kwargs):
        """Loads the raw data of the file into a DataFrame and gives common columns (time, current, voltage, etc.) a standardized name.

        Args:
            filepath (str | Pathlike): The file to read.

        Returns:
            pandas.DataFrame: A DataFrame containing the file's raw data with standardized column names.
        """
        pass





class SimpleFileInterface(FileInterface):
    """A FileInterface template for when standardizing the data consists of renaming and doing simple operations on columns."""



    @classmethod
    @abstractmethod
    def column_map(cls):
        """Returns a dictionary mapping standardized column names such as Time (s) to filetype specific column names such as time/s in the case of .mpt files."""
        pass


    @classmethod
    @abstractmethod
    def column_operations_map(cls):
        """Returns a dictionary mapping filetype specific column names to functions to apply to the column when standardizing the data."""
        pass


    @classmethod
    @abstractmethod
    def columns_to_zero(cls):
        """Returns a list containing standardized column names to offset such that the first value is 0 when zerocols='all' in filepath_to_standard_data_df."""
        pass




    @classmethod
    def filepath_to_standard_data_df(cls, filepath, usecols=None, zerocols='all', **kwargs):
        """Loads the raw data of the file into a DataFrame and gives common columns (time, current, voltage, etc.) a standardized name.

        Args:
            filepath (str | Pathlike): The file to read.
            usecols (None | list[str], optional): Specifies which columns to load. Uses standardized column names. Defaults to 'all'.
            zerocols (None | list[str] | str, optional): Specifies which columns to offset such that the first value is 0. Uses standardized column names. Defaults to None.
            **kwargs: Additional keyword arguments passed to SimpleFileInterface.filepath_to_raw_data_df.

        Returns:
            pandas.DataFrame: A DataFrame containing the file's raw data with standardized column names.
        """

        column_keyword_map = cls.column_keyword_map()
        inverse_column_keyword_map = {v : k for k, v in column_keyword_map.items()}
        operations_map = cls.column_operations_map()

        # Convert usecols to raw names
        if usecols is not None:
            usecols_raw = [column_keyword_map[column] if column in column_keyword_map else column for column in usecols]
        else:
            usecols_raw = None

        # Load raw data
        raw_data_df = cls.filepath_to_raw_data_df(filepath, usecols=usecols_raw, **kwargs)

        # Apply column operations
        standard_data_df = pd.DataFrame.from_dict({(inverse_column_keyword_map[column] if column in inverse_column_keyword_map else column) : (operations_map[column](raw_data_df[column]) if column in operations_map else raw_data_df[column]) for column in raw_data_df.columns})

        # Zero specified columns
        if zerocols is not None:
            if zerocols == 'all':
                zerocols = cls.columns_to_zero()
            for column in set(zerocols).intersection(set(standard_data_df.columns)):
                standard_data_df[column] = standard_data_df[column] - standard_data_df[column].iloc[0]
        
        return standard_data_df