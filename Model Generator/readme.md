# Model Generator
The model generator finds the best suitable model that satisfies the user-specific requirements. It can merge mutliple datasets together and compare multiple models to find the best one.

## How To

1. Add the requirements to the [user_specific_requirements.yaml](user_specific_requirements.yaml) file.
1. [Dataset pre-processing readme](Dataset_preprocessing/readme.md): configure the settings in [dataset input](Dataset_preprocessing/dataset_input_config.yaml) and run the [dataset_preprocessing.py](dataset_preprocessing.py) file.
2. [Hyper-parameter optimization readme](Hyper_parameter_optimization/readme.md): configure the settings in the [config](Hyper_parameter_optimization/hyper_parameter_optimization_config.yaml) and run the [hyper_parameter_optimization.py](hyper_parameter_optimization.py) file.
3. [Model selection readme](Model_selection/readme.md): configure the settings in the [config](Model_selection/model_selection_config.yaml) and run the [model_selection.py](model_selection.py) file.
4. [Model conversion readme](Model_conversion/readme.md): configure the settings in the [config](Model_conversion/model_conversion_config.yaml) and run the [model_conversion.py](model_conversion.py) file.
