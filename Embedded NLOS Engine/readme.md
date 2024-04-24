# Embedded NLOS Engine

The Embedded NLOS Engine is a self-contained implementation in C that allows for the execution of models produced with the model generator tool. 
- ngine.h and ngine.c:  The interface for the embedded NLOS Engine is defined in the ngine.h and ngine.c 
- models/xgboost/xgb_ngine_impl.c: The interface implementation for XGBoost models can be found in the models/xgboost/xgb_ngine_impl.c file
- model/xgboost/{xgb_classifier.h, xgb_regressor.h, xgb_model_config_output.h}: The XGBoost models and their configurations are contained in the model/xgboost/{xgb_classifier.h, xgb_regressor.h, xgb_model_config_output.h} files, which are generated using the *model generator* tool.

The embedded NLOS engine is executed by executing the *handler (ngine_handle)* as follows:
```
ngine_result_t result;
ngine_request_t ngine_request = {
    .cir={1,2,3,4,...} // CIR samples
    .cmd=NGINE_CLASSIFICATION_AND_REGRESSION, //Perform classification and regression at once
    .pacc=128, // PAC number
};
ngine_handle(&ngine_request,&result);
```
The *result* struct stores the classification in *result.NLOS* and error correction in *result.predicted_error*.