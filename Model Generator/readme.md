# Model Generator
The model generator finds the best suitable model that satisfies the user-specific requirements and converts it to C code that the Embedded NLOS Engine can use. It can merge mutliple datasets together and compare different models to find the best suitable model for the requirements.

## How To

1. Add the requirements to the [user_specific_requirements.yaml](user_specific_requirements.yaml) file.
2. [Dataset pre-processing readme](Dataset_preprocessing/readme.md): configure the settings in [dataset input](Dataset_preprocessing/dataset_input_config.yaml) and run the [dataset_preprocessing.py](dataset_preprocessing.py) file.
3. [Hyper-parameter optimization readme](Hyper_parameter_optimization/readme.md): configure the settings in the [config](Hyper_parameter_optimization/hyper_parameter_optimization_config.yaml) and run the [hyper_parameter_optimization.py](hyper_parameter_optimization.py) file.
4. [Model selection readme](Model_selection/readme.md): configure the settings in the [config](Model_selection/model_selection_config.yaml) and run the [model_selection.py](model_selection.py) file.
5. [Model conversion readme](Model_conversion/readme.md): configure the settings in the [config](Model_conversion/model_conversion_config.yaml) and run the [model_conversion.py](model_conversion.py) file.
6. Move the model and pre-processing files, found in the model conversion output, to the ``Embedded NLOS Engine``.
