# Model Selection

This component calculates the Pareto Front to quickly filter out unsuitable models. The an achievement scalarization function (ASF) is used to find the best suitable models. Note that models that either classify or perform regression are simply return the best accuracy or R2-score. However, the ASF function finds the best trade-off between both objectives and is used for models that provide both classification and regression simultaneously.
## How To
Use the ``model_selection_config.yaml`` file to add which objective functions should be optimized and add the file paths 

```
---
# Objective functions to select best model
objective_class: "score_accuracy"
objective_reg: "score_r2"
...
```