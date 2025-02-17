# Electrical Test Analyzer

Electrical Test Analyzer is a collection of modules for loading and analyzing electrical test data.


## Requirements


**Cell parameters log**: A file or files (.csv, .xlsx, etc.) containing information about cell builds. Contains one row per sample. Should at least contain columns "Sample name" and "Active area (cm2)". Other example columns include Cell type, Cathode type, Separator type, Separator thickness (mm), etc.

**Test parameters log**: Files (.csv, .xlsx, etc.) containing information about tests. Contains one row per test. Should at least contain columns "Sample name" and "File specifier". "File specifier" should contain a substring of the raw data filename such that, along with the sample name, a unique file in the data directory is specified (i.e. `sample_name is in file_name and file_specifier is in file_name` is `True` for exactly one file in the data directory). Note that "File specifier" can simply be the entire file name. Other example columns include Cell temperature (C), Bubbler type, Bubbler temperature (C), Gas type, etc.

**Data directory**: A directory or directories containing the raw data files.

An example file structure is as follows:
```
├──logs/
│   ├──cell_parameters.csv
│   ├──test_parameters/
│   │   ├──GITT_parameters.csv
│   │   ├──constant_current_parameters.csv
│   │   └── ...
│   └──component_properties.csv
└──data_directory/
```

## Usage

An electrical test analyzer workflow involves three major steps: loading the data, analyzing the data, and plotting or saving the data.

### Loading the data

Data is queried using QueryManager and FilterSet objects. The QueryManager is initialized with cell parameter and test parameter filepaths. Upon initialization, all tests in the test parameter files are selected. At any point, the number of selected tests can be checked with `QueryManager.num_tests`

```python
from electrical_test_analyzer import QueryManager

qm = QueryManager(cell_parameters_filepath, test_parameters_filepath)

print(f'{qm.num_tests} tests selected')

22 tests selected
```

The data can then be filtered by any column in the cell and test parameters using the FilterSet objects. A FilterSet is composed of a combination of the following filters, which will be applied in conjunction.

| Filter Type          | Expected Format                        | Example Usage                                     |
|----------------------|----------------------------------------|---------------------------------------------------|
| `query_filters`      | `list` of strings (pandas `.query()`)  | `["Cell temperature (C) > 110", "Bubbler temperature (C) < 85"]`|
| `isin_filters`       | `dict` where keys are column names, values are lists of allowed values | `{"Sample name": ["NCC001AB-EC00-01", "NHC000AK-EC00-01"]}`|
| `notin_filters`      | `dict` where keys are column names, values are lists of disallowed values | `{"Cell type": ["Flow"]}`|
| `between_filters`    | `dict` where keys are column names, values are `(low, high)` tuples | `{"Active area (cm2)": (0.5, 1.0)}`|
| `lambda_filters`     | `list` of functions returning boolean masks | `[lambda df: df["Bubbler temperature (C)"] % 2 == 0]`|
| `null_check_filters` | `dict` where keys are column names, values are `True` (keep nulls) or `False` (drop nulls) | `{"Active area (cm2)": False}`|


The following FilterSets will keep sample NCC001AB-EC00-01 and all flow cells:
```python
from electrical_test_analyzer import FilterSet

sample_names = ['NCC001AB-EC00-01']
samplename_filterset = FilterSet(isin_filters={'Sample name' : sample_names})

cell_types = ['Flow']
celltype_filterset = FilterSet(isin_filters{'Cell type' : cell_types})
```

The FilterSets are added and applied to the QueryManager as follows. Adding multiple FilterSets to the QueryManager will yield the union of their results.

```python
qm.add_filterset([samplename_filterset, celltype_filterset])
qm.apply_filtersets()

print(f'{qm.num_tests} tests selected')

4 tests selected
```

Finally, the QueryManager is pointed to a directory or set of directories to recursively search. For each `filename` with a valid extension (.mpt), the filepath is associated with the a test if the following condition is satisfied using the sample name and file specifier of the test:`sample_name is in filename and file_specifier is in filename`. The search outputs a pandas DataFrame which contains a row for each test and columns for each cell and test parameter, as well as a `filepath` column containing the associated filepath.

```
query_df = qm.search(data_directory)
````

### Analyzing the data

In general, analyzers take in DataFrames and output DataFrames via an `analyze()` method. The following analyzers take in the query DataFrame from `QueryManager.search()` and output results as well as the cell and test parameters. Any columns with area-normalizable units (mA, mAh, W, Ohm, etc.) are copied into an area-normalized version (mA/cm2, mAh/cm2, W/cm2, Ohm-cm2, etc.): using the 'Active area (cm2)' column.

**RawDataLoader**: Returns the raw data from each test. Note that certain standard columns are renamed.

**GITTAnalyzer**: Returns metrics for each pair of consecutive polarize, rest sequences in each test.

**EISLoader**: Returns raw EIS data from each scan in each test.

**EISBasicAnalyzer**: Returns the minimum first-quadrant Re(Z) value along with some other metrics for each EIS scan within each test.

**MetricAnalyzer**: Returns total time, capacity, and energy for each test along with energy and power density metrics. Calculated for the total test and considering only the polarization steps. Requires the cell parameter log to contain `Cathode type`, `Cathode thickness (mm)`, `Separator type`, `Separator thickness (mm)`, `Anode type`, and `Anode thickness (mm)` columns, as well as a separate component properties log with densities for each material.

The following example shows the metrics calculated by the GITTAnalyzer:
```python
from electrical_test_analyzer import GITTAnalyzer

gitt_output_df = GITTAnalyzer.analyze(query_df)

print(gitt_output_df.columns)

Index(['Sample name', 'File specifier', 'Cell temperature (C)', 'Bubbler type',
       'Bubbler temperature (C)', 'Bubbler wt NaOH', 'Gas type',
       'Flow rate (cfs)', 'Oven', 'Batch', 'Cell type', 'Active area (cm2)',
       'Cathode type', 'Cathode thickness (mm)', 'Cathode geometry',
       'Separator type', 'Separator thickness (mm)', 'Anode type',
       'Anode thickness (mm)', 'filepath', 'Pulse count',
       'Polarize start time (h)', 'Polarize end time (h)',
       'Rest start time (h)', 'Rest end time (h)',
       'Polarize start capacity (mAh)', 'Polarize end capacity (mAh)',
       'Mean polarization current (mA)',
       'Standard deviation polarization current (mA)',
       'Median polarization current (mA)',
       'Mean last 10 polarization current points (mA)',
       'Mean polarization voltage (V)',
       'Standard deviation polarization voltage (V)',
       'Median polarization voltage (V)',
       'Mean last 10 polarization voltage points (V)', 'Mean rest voltage (V)',
       'Standard deviation rest voltage (V)', 'Median rest voltage (V)',
       'Mean last 10 rest voltage points (V)',
       'Mean last 10 DC resistance (Ohm)', 'Polarize start capacity (mAh/cm2)',
       'Polarize end capacity (mAh/cm2)', 'Mean polarization current (mA/cm2)',
       'Standard deviation polarization current (mA/cm2)',
       'Median polarization current (mA/cm2)',
       'Mean last 10 polarization current points (mA/cm2)',
       'Mean last 10 DC resistance (Ohm-cm2)'],
      dtype='object')
```

### Plotting/saving the data

Plotly Express provides functions for plotting directly from pandas DataFrames. Electrical Test Analyzer provides a basic version in `plot_df`. Pandas provides methods for writing DataFrames to disk.