
#include <string.h>
#include <assert.h>
#include "../../ngine.h"
#include "xgb_model_config_output.h"
#include "xgb_classifier.h"
#include "xgb_regressor.h"

float ngine_cir_buffer[NGINE_CIR_LEN];
float ngine_cir_buffer_regression[XGB_REGRESIION_CIR_LEN];
float ngine_cir_buffer_classifcation[XGB_CLASSIFICATION_CIR_LEN];
float range;


void ngine_scale_pacc(float *cir_data, uint16_t number_samples, uint16_t pacc)
{
    for(uint16_t i=0; i<number_samples;i++)
        cir_data[i]   = cir_data[i]/pacc;
    
}
void ngine_scale(float *cir_data, float* scale_x_min, float* scale_x_max, uint16_t number_samples) 
{
   
    for(uint16_t i=0; i<number_samples;i++)
    {
       cir_data[i] = (cir_data[i] - scale_x_min[i])/(scale_x_max[i]-scale_x_min[i]);
    }
}

/*-----------------------------------------------------------------------*/

ngine_status_t ngine_preprocessing(ngine_request_t *cir_data)
{

    memcpy((void*)ngine_cir_buffer, cir_data->cir, sizeof(cir_data->cir));
    ngine_scale_pacc(ngine_cir_buffer,sizeof(ngine_cir_buffer)/sizeof(float),cir_data->pacc);

    if(XGB_CLASSIFICATION_CIR_START < XGB_REGRESIION_CIR_START)
    {
        memcpy((void*)ngine_cir_buffer_classifcation, (void*)&ngine_cir_buffer[0], sizeof(ngine_cir_buffer_classifcation));
        memcpy((void*)ngine_cir_buffer_regression, (void*)&ngine_cir_buffer[XGB_REGRESIION_CIR_START-XGB_CLASSIFICATION_CIR_START], sizeof(ngine_cir_buffer_regression)); 
    }
    else if(XGB_REGRESIION_CIR_START < XGB_CLASSIFICATION_CIR_START)
    {
        memcpy((void*)ngine_cir_buffer_regression, (void*)&ngine_cir_buffer[0], sizeof(ngine_cir_buffer_regression));
        memcpy((void*)ngine_cir_buffer_classifcation, (void*)&ngine_cir_buffer[XGB_CLASSIFICATION_CIR_START-XGB_REGRESIION_CIR_START], sizeof(ngine_cir_buffer_classifcation)); 
    }
    else 
    {
        memcpy((void*)ngine_cir_buffer_regression, (void*)&ngine_cir_buffer[0], sizeof(ngine_cir_buffer_regression));
        memcpy((void*)ngine_cir_buffer_classifcation, (void*)&ngine_cir_buffer[0], sizeof(ngine_cir_buffer_classifcation)); 
    }
    ngine_scale(ngine_cir_buffer_classifcation,class_scale_x_min,class_scale_x_max,sizeof(ngine_cir_buffer_classifcation)/sizeof(float));
    ngine_scale(ngine_cir_buffer_regression,reg_scale_x_min,reg_scale_x_max,sizeof(ngine_cir_buffer_regression)/sizeof(float));

    return OK;
}

ngine_status_t ngine_classification(NLOS_status_t *nlos)
{
    int class_result = xgb_class_predict(ngine_cir_buffer_classifcation);
    if(class_result==1) 
        *nlos = NLOS;
    else
        *nlos = LOS;
    return OK;
}

ngine_status_t ngine_regression(float *predicted_error)
{
    *predicted_error = xgb_reg_predict(ngine_cir_buffer_regression);   
    *predicted_error = (*predicted_error)*(reg_scale_y_max - reg_scale_y_min) + reg_scale_y_min;

    return OK;
}
