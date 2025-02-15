import os

from electrical_test_analyzer.dataframe_operations import naive_df_area_normalization
from electrical_test_analyzer.file_interfaces import file_interface_extension_map
from electrical_test_analyzer.test_analyzers.test_analyzer_template import FileAnalyzer


class RawDataLoader(FileAnalyzer):



    @classmethod
    def filepath_to_df(cls, filepath, interface=None, **kwargs):
        _, ext = os.path.splitext(filepath)
        interface = interface or file_interface_extension_map[ext]
        results_df = interface.filepath_to_standard_data_df(filepath, **kwargs)
        return results_df
    


    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):
        output_df = super().analyze(input_df, concat_kwargs=concat_kwargs, **kwargs)
        output_df = naive_df_area_normalization(output_df)
        return output_df

