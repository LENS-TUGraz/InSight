# Dataset Pre-processing

The dataset pre-processing component is used to load one or more datasets, pre-process them to a common format and mix and merge the datasets together. Note that multiple datasets should have common column names and the columns should use the same dimensions and scaling. For easier compatibility, we propose to use the measured range column as ``rng``, the real (or true) range column as ``real_rng``, and the channel impulse response samples as ``cir``.

## How To 
Use the ``dataset_input_config.yaml`` file to enter the following information: 

```
# DATASETS
# Choose which datasets should be merged together by putting them in the same list.
datasets_to_run:
    train: ["stocker_unf"] 
    test: ["stocker_unf"]
    
    
# Pipeline for initial data post processing, e.g., calculate global features ###
pre_processing:
    # define start and end of CIR in relation to the detected first path
    cir_start: 5
    cir_end: 30
    
    # Confine data if necessary
    max_rng: 10 
    min_rng: -0.5 
    max_rng_e: 10 
    min_rng_e: -0.8 
    
    # Use none, use given, or recreate bias correction using polynomial function
    bias_correction: "poly" # none, given, poly
    
    # Data Cleaning to filter outliers
    filter_negative_values_column: null
    outlier_column: "rng"
    groupby_columns: ["setup_id"]
    filter_outlier_verbose: false
```

The datasets to run are defined in the ``dataloaders`` folder where new datasets can be added. Make sure that the datasets have a common format: i.e., have the same CIR length, are scaled the same way, first path is aligned to a common position, ... . Note that we take care to keep similar measurements together (e.g., 50 measurements from a single node placement) to not bias our results by having similar measurements in the training and testing dataset. We mark these measurements with the same ``setup_id`` and this id is later used when the datasets are mixed or if we perform a k-fold to split training and testing sets.

The pre-processing pipeline defines a common pre-processing for the datasets: i.e., which portion of the CIR is taken in relation to the detected first path, confine range measurements to a common area, apply a bias correction or not, and whether to filter outliers.