#ifndef _XGBoost_H
#define _XGBoost_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

uint8_t do_regression;
uint8_t do_classification;

int xgb_class_predict(float* x);
float xgb_reg_predict(float* x);

#endif /* _XGBoost_H */