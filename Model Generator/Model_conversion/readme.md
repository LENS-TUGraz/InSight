# Model Conversion

This component converts the python models to C-code. It produces the model files themselves and the config files required to process the CIR. These files can be placed in the [Embedded NLOS Engine](../Embedded NLOS Engine/readme.md).
## How To

Use the ``model_conversion_config.yaml`` file to enter the model and type that should be converted. The model will be retrained with the hyper-parameters provided by [hyper-parameter optimization](Hyper_parameter_optimization/readme.md). Optionally the model can be tested again to check that everything went smoothly.

```
---
model_to_run:
    model: "xgboost"
    mltype: ["class", "reg"]
test_model: True                   
use_kfold: False
use_validation_set: True                                        
fp_pos: 20
...
```