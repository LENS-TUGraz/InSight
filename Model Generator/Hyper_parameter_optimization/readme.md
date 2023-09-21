# Hyper-Parameter Optimization

The
## How To
Use the ``hyper_parameter_optimization_config.yaml`` file to add which ML models is used and if classification or regression should be performed. Also the training configuration and grid search configuration can be placed here as shown in this example:

```
---
# ML models to run
# Choose which ML models should be used in the hyper parameter optimization and choos classification, regression, or mo for multi-output models.
models_to_run: [['xgboost', 'class'],  ['xgboost', 'reg']]


# Hyper Parameter Optimization Config
training_configuration:
    kfolds: 5
    kfgroups: ["setup_id"] # Split KFold by setup_id 
    conditions: ['los', "wlos",'nlos'] # Only consider los and nlos labels #'wlos', #'nlos'
    resample: False # Equalize the number of smaples of NLOS field (currently onls 1, 0 ) 
    resample_tag: 'NLOS' # if 'resample=True' then resample based on which tag
    early_stopping: True

grid_search_config:
    xgboost:
        cir_start: [0, 5, 10, 15, 18, 20, 25]
        cir_end: [172, 132, 72, 38, 25, 20, 15]
        max_depth: [10, 6, 3, 1]
        n_estimators: [100, 60, 30, 10]
```

The training configuration defines if and how many k-folds should be applied and by which id the data should be split. As mentioned in the [Dataset Preprocessing](Dataset_preprocessing/readme.md), this id should make sure that similar measurements from an identical node placement are not in the train and test dataset. If the dataset is unevenly balanced with line-of-sight and non-line-of-sight measurements, use the resampling options the create a balanced set.

The grid search configuration defines the different hyper-parameters that should be swept through.