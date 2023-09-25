#ifndef ENGINE_H
#define ENGINE_H 
#include <stdint.h>
#include <stdio.h>
#include "ngine_config.h"


#ifdef ML_MODEL_ENABLE_XGBOOST
#include "models/xgboost/xgb_model_config_output.h"
#endif


#if REGRESIION_CIR_FP_POS != CLASSIFICATION_CIR_FP_POS
#error "CLASSIFICATION_CIR_FP_POS and REGRESIION_CIR_FP_POS must be the same"
#endif
#if REGRESIION_CIR_FP_POS != 20
#error "CLASSIFICATION_CIR_FP_POS and REGRESIION_CIR_FP_POS must be set to 20"
#endif


typedef enum {LOS=0, NLOS=1, LOS_STATUS_NONE} NLOS_status_t;
typedef enum {NOT_IMPLEMENTED=-1,OK=0} ngine_status_t;

typedef enum {
    NGINE_NONE=0,
    NGINE_CLASSIFICATION=1,
    NGINE_REGRESSION,
    NGINE_CLASSIFICATION_AND_REGRESSION,
    NGINE_REGRESSION_IF_NLOS
} ngine_cmd_t;

typedef struct {
    ngine_cmd_t result_type;
    NLOS_status_t NLOS;
    float predicted_error;
} ngine_result_t;

typedef struct {
    float cir[NGINE_CIR_LEN];
    float range;
    uint16_t pacc;
    ngine_cmd_t cmd;
} ngine_request_t;


ngine_status_t ngine_handle(ngine_request_t *cir_data, ngine_result_t *result);
//ngine_status_t init_engine(float *cir, float range);
ngine_status_t ngine_preprocessing(ngine_request_t *cir_data);
ngine_status_t ngine_classification(NLOS_status_t *nlos);
ngine_status_t ngine_regression(float *predicted_error);
//ngine_status_t multi_output(cir_t *cir);
#endif