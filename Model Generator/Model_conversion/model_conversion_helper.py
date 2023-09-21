    
def generate_cir_config_macro():
    string = """
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
    """
    return string

def generate_cir_config_header(class_cir_start=-1, class_cir_end=-1, cir_fp_pos=-1, reg_cir_start=-1, reg_cir_end=-1):
    string = f"""
#define XGB_CLASSIFICATION_CIR_START {class_cir_start}
#define XGB_CLASSIFICATION_CIR_END {class_cir_end}
#define XGB_CLASSIFICATION_CIR_FP_POS {cir_fp_pos}
#define XGB_CLASSIFICATION_CIR_LEN XGB_CLASSIFICATION_CIR_END - XGB_CLASSIFICATION_CIR_START

#define XGB_REGRESIION_CIR_START {reg_cir_start}
#define XGB_REGRESIION_CIR_END {reg_cir_end}
#define XGB_REGRESIION_CIR_FP_POS {cir_fp_pos}
#define XGB_REGRESIION_CIR_LEN XGB_REGRESIION_CIR_END - XGB_REGRESIION_CIR_START
    """
    
    
    string = string + """
#define CLASSIFICATION_CIR_START XGB_CLASSIFICATION_CIR_START
#define CLASSIFICATION_CIR_END XGB_CLASSIFICATION_CIR_END
#define CLASSIFICATION_CIR_FP_POS XGB_CLASSIFICATION_CIR_FP_POS
#define REGRESIION_CIR_START XGB_REGRESIION_CIR_START
#define REGRESIION_CIR_END XGB_REGRESIION_CIR_END
#define REGRESIION_CIR_FP_POS XGB_REGRESIION_CIR_FP_POS

#define NGINE_CIR_START MIN(CLASSIFICATION_CIR_START,REGRESIION_CIR_START)
#define NGINE_CIR_END MAX(CLASSIFICATION_CIR_END, REGRESIION_CIR_END)
#define NGINE_CIR_FP XGB_REGRESIION_CIR_FP_POS
#define NGINE_CIR_LEN NGINE_CIR_END - NGINE_CIR_START
    """
    return string


def generate_scaling_parameter_config_header(class_scale_len=-1, reg_scale_len=-1):
    string = f"""
/*---------------------------------------------*/
/*---------Scaling parameter definition--------*/
/*---------------------------------------------*/
extern float reg_scale_x_min[{reg_scale_len}];
extern float reg_scale_x_max[{reg_scale_len}]; 
extern float reg_scale_y_min;
extern float reg_scale_y_max;

extern float class_scale_x_max[{class_scale_len}]; 
extern float class_scale_x_min[{class_scale_len}];
    """
    return string

def generate_scaling_parameter_code(class_scale_x_min=[-1], class_scale_x_max=[-1], reg_scale_x_min=[-1], reg_scale_x_max=[-1], reg_scale_y_min=-1, reg_scale_y_max=-1):
    string = f"""
float reg_scale_x_min[{len(reg_scale_x_min)}] = {{{','.join([str(e) for e in reg_scale_x_min])}}};
float reg_scale_x_max[{len(reg_scale_x_max)}] = {{{','.join([str(e) for e in reg_scale_x_max])}}};
float reg_scale_y_min = {reg_scale_y_min};
float reg_scale_y_max = {reg_scale_y_max};

float class_scale_x_min[{len(class_scale_x_min)}] = {{{','.join([str(e) for e in class_scale_x_min])}}};
float class_scale_x_max[{len(class_scale_x_max)}] = {{{','.join([str(e) for e in class_scale_x_max])}}};
    """
    return string

def generate_scaling_parameter_code_mo(class_scale_x_min=[-1], class_scale_x_max=[-1], reg_scale_x_min=[-1], reg_scale_x_max=[-1], reg_scale_y_min=-1, reg_scale_y_max=-1):
    string = f"""
float reg_scale_x_min[1] = {{{reg_scale_x_min}}};
float reg_scale_x_max[1] = {{{reg_scale_x_max}}};
float reg_scale_y_min = {reg_scale_y_min};
float reg_scale_y_max = {reg_scale_y_max};

float class_scale_x_min[1] = {{{class_scale_x_min}}};
float class_scale_x_max[1] = {{{class_scale_x_max}}};
    """
    return string