from abc import ABC, abstractmethod

import pandas as pd





class TestAnalyzer(ABC):



    default_concat_kwargs = dict(ignore_index=True)



    @classmethod
    @abstractmethod
    def split_df(cls, input_df, **kwargs):
        pass



    @classmethod
    @abstractmethod
    def analyze_subset_df(cls, *subset_df, **kwargs):
        pass



    @classmethod
    def analyze(cls, input_df, concat_kwargs=None, **kwargs):

        concat_kwargs = concat_kwargs or {}
        concat_kwargs = {**cls.default_concat_kwargs, **concat_kwargs}

        output_df = pd.concat([analyzed_subset for subset_df in cls.split_df(input_df, **kwargs) if (analyzed_subset := cls.analyze_subset_df(subset_df, **kwargs)) is not None], **concat_kwargs)

        return output_df





class FileAnalyzer(TestAnalyzer):



    @classmethod
    @abstractmethod
    def filepath_to_df(cls, filepath, **kwargs):
        pass



    @classmethod
    def split_df(cls, input_df, **kwargs):
        return [row for _, row in input_df.iterrows()]



    @classmethod
    def analyze_subset_df(cls, row, **kwargs):
        results_subset_df = cls.filepath_to_df(row['filepath'], **kwargs)
        repeated_row_df = pd.DataFrame(len(results_subset_df) * [row]).reset_index(drop=True) # Index must be reset and drop=True for concatenation to work
        output_subset_df = pd.concat((repeated_row_df, results_subset_df), axis=1)
        return output_subset_df

        