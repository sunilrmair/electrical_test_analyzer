import re

import numpy as np
import pandas as pd


from electrical_test_analyzer.file_interfaces.file_interface_template import SimpleFileInterface





class MPTInterface(SimpleFileInterface):
    """A collection of methods for loading and manipulating data from EC-Lab .mpt files."""



    CHECK_HEADER_LINES = 10
    CHECK_HEADER_PHRASE = 'Nb header lines'
    DELIMITER = '\t'

    mode_dict = {1 : 'POLARIZE', 2 : 'EIS', 3 : 'REST'}



    @classmethod
    def get_header_line_number(cls, filepath, encoding):
        """Reads the beginning CHECK_HEADER_LINES of the file looking for CHECK_HEADER_PHRASE to determine the line number of the header. Helper function for MPTInterface.filepath_to_raw_data_df.

        Args:
            filepath (str | Pathlike): The file to read.
            encoding (str): Encoding used to open the file.

        Returns:
            int: The line number of the header. Defaults to 1 if CHECK_HEADER_PHRASE not found in CHECK_HEADER_LINES.
        """

        with open(filepath, 'r', encoding=encoding) as file:
            for _ in range(cls.CHECK_HEADER_LINES):
                line = next(file, None)
                if line is None:
                    break
                line_stripped = line.strip()
                if line_stripped[:len(cls.CHECK_HEADER_PHRASE)] == cls.CHECK_HEADER_PHRASE:
                    return int(re.search(cls.CHECK_HEADER_PHRASE + r'\s*:\s*(\d+)\b', line_stripped).group(1))
        return 1
    


    @classmethod
    def filepath_to_raw_data_df(cls, filepath, **kwargs):
        """Loads the raw data of the file into a DataFrame.

        Args:
            filepath (str | Pathlike): The file to read.
            **kwargs: Additional keyword arguments passed to pandas.read_csv.

        Returns:
            pandas.DataFrame: A DataFrame containing the file's raw data.
        """
        
        encoding = kwargs.get('encoding', 'latin-1')

        header_line_number = cls.get_header_line_number(filepath, encoding)

        default_kwargs = dict(delimiter=cls.DELIMITER, skiprows=header_line_number - 1, encoding=encoding)
        kwargs = {**default_kwargs, **kwargs}

        raw_data_df = pd.read_csv(filepath, **kwargs)

        return raw_data_df
    


    @classmethod
    def column_keyword_map(cls):
        """Returns a dictionary mapping standardized column names to filetype specific column names.

        Returns:
            dict: a dictionary mapping standardized column names to filetype specific column names.
        """

        keyword_map = {
            'Time (s)' : 'time/s',
            'Capacity (mAh)' : 'Capacity/mA.h',
            'Voltage (V)' : 'Ewe/V',
            'Current (mA)' : 'I/mA',
            'Power (W)' : 'P/W',
            'Average Current (mA)' : '<I>/mA',
            'Energy (Wh)' : '|Energy|/W.h',
            'Re Z (Ohm)' : 'Re(Z)/Ohm',
            '-Im Z (Ohm)' : '-Im(Z)/Ohm',
            'Frequency (Hz)' : 'freq/Hz'
        }

        return keyword_map
    


    @classmethod
    def column_operations_map(cls):
        """Returns a dictionary mapping filetype specific column names to functions to apply to the column when standardizing the data.

        Returns:
            dict: A dictionary mapping filetype specific column names to functions to apply to the column when standardizing the data. Functions should take pandas.Series as their argument.
        """

        # Identity function can be omitted, only here as example
        operations_map = {
            'time/s' : lambda x : x,
        }

        return operations_map
    


    @classmethod
    def columns_to_zero(cls):
        """Returns a list containing standardized column names to offset such that the first value is 0 when zerocols='all' in filepath_to_standard_data_df.

        Returns:
            list[str]: A list containing standardized column names to offset such that the first value is 0 when zerocols='all' in filepath_to_standard_data_df
        """

        columns_to_zero = [
            'Time (s)',
            'Capacity (mAh)',
            'Energy (Wh)'
        ]

        return columns_to_zero
    
    

    @classmethod
    def filepath_to_standard_data_df(cls, filepath, usecols=None, zerocols='all', **kwargs):

        required_cols = ['mode']
        if usecols is not None:
            usecols.extend([col for col in required_cols if col not in usecols])
            
        standard_data_df = super().filepath_to_standard_data_df(filepath, usecols=usecols, zerocols=zerocols, **kwargs)

        # If frequency data is included, use to refine classification of mode
        if 'Frequency (Hz)' in standard_data_df.columns:
            standard_data_df['MODE'] = np.where(standard_data_df['Frequency (Hz)'] != 0, 'EIS', standard_data_df['mode'].map(cls.mode_dict))
        else:
            standard_data_df['MODE'] = standard_data_df['mode'].map(cls.mode_dict)
        return standard_data_df






class MPTThreeInterface(MPTInterface):
    """A collection of methods for loading and manipulating data from EC-Lab .mpt files for three electrode experiments."""

    @classmethod
    def column_keyword_map(cls):
        """Returns a dictionary mapping standardized column names to filetype specific column names.

        Returns:
            dict: a dictionary mapping standardized column names to filetype specific column names.
        """

        keyword_map = {
            'Time (s)' : 'time/s',
            'Capacity (mAh)' : 'Capacity/mA.h',
            'Voltage (V)' : 'Ewe-Ece/V',
            'Working Voltage (V)' : 'Ewe/V',
            'Counter Voltage (V)' : 'Ece/V',
            'Current (mA)' : 'I/mA',
            'Average Current (mA)' : '<I>/mA',
            'Energy (Wh)' : 'Energy we discharge/W.h',
            'Re Z (Ohm)' : 'Re(Z)/Ohm',
            '-Im Z (Ohm)' : '-Im(Z)/Ohm',
            'Frequency (Hz)' : 'freq/Hz'
        }

        return keyword_map
    


    @classmethod
    def column_operations_map(cls):
        """Returns a dictionary mapping filetype specific column names to functions to apply to the column when standardizing the data.

        Returns:
            dict: A dictionary mapping filetype specific column names to functions to apply to the column when standardizing the data. Functions should take pandas.Series as their argument.
        """


        # Identity function can be omitted, only here as example
        operations_map = {
            'time/s' : lambda x : x,
            'Capacity/mA.h' : lambda x : x,
        }

        return operations_map
    


    @classmethod
    def columns_to_zero(cls):
        """Returns a list containing standardized column names to offset such that the first value is 0 when zerocols='all' in filepath_to_standard_data_df.

        Returns:
            list[str]: A list containing standardized column names to offset such that the first value is 0 when zerocols='all' in filepath_to_standard_data_df
        """

        columns_to_zero = [
            'Time (s)',
            'Capacity (mAh)',
            'Energy (Wh)'
        ]

        return columns_to_zero