# Electrical Test Analyzer

Electrical Test Analyzer is a collection of modules for loading and analyzing electrical test data.


## Requirements

Cell parameters log: A file or files (.csv, .xlsx, etc.) containing information about cell builds. Should at least contain columns "Sample name" and "Active area (cm2)". Other example columns include: [TODO: list]

Test parameters log: Files (.csv, .xlsx, etc.) containing information about tests. Should at least contain columns "Sample name" and "File specifier". "File specifier" should contain a substring of the raw data filename such that, along with the sample name, a unique file in the data directory is specified (i.e. `sample_name is in file_name and file_specifier is in file_name` is `True` for exactly one file in the data directory). Note that "File specifier" can simply be the entire file name. Other example columns include [TODO: list].

Data directory: A directory or directories containing the raw data files.


## Usage

QueryManager