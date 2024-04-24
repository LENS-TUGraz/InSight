# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:56:09 2022

@author: turtle9
"""


def libsvm_converter(file_name, model, suffix, n_features):
    param = model.param
    l = model.l
    sv_coef = model.get_sv_coef()
    SV = model.get_SV()
    nr_class = model.get_nr_class()
    
    f = open(f"{file_name}.c", "w")

    f.write("#include \"svm.h\"\n\n")
    f.write(f"const struct svm_parameter param{suffix} = {{\n")
    f.write(f".svm_type = {param.svm_type},\n")
    f.write(f".kernel_type = {param.kernel_type},\n")
    f.write(f".degree = {param.degree},\n")
    f.write(f".gamma = {param.gamma:.3f},\n")
    f.write(f".coef0 = {param.coef0:.3f},\n")
    f.write(f"}};\n\n")

    f.write(f"int nr_class{suffix} = {nr_class};\n\n")
    f.write(f"int l{suffix} = {l};\n\n")

    #SV
    f.write(f"struct svm_node SV{suffix}[{l}][{n_features + 1}] = {{")
    for i in range(l):
        p = SV[i]
        f.write("{")
        for idx, (key, value) in enumerate(p.items()):
            f.write(f"{{{key}, {value:.3f}}}")
            if idx != len(p) - 1:
                f.write(", ")
        if i != (l - 1):
            f.write("}, ")
            
    f.write("}};\n\n")
    # sv_coef
    f.write(f"float sv_coef{suffix}[{nr_class - 1}][{l}] = {{")
    for i in range(nr_class - 1):
        f.write("{")
        for j in range(l):
            if(j < l - 1):
                f.write(f"{sv_coef[j][i]:.3f}, ")
            else:
                f.write(f"{sv_coef[j][i]:.3f}")
        if(i < nr_class-2):
            f.write("},")
        else:
            f.write("}};\n\n")
    # rho
    f.write(f"float rho{suffix}[{nr_class - 1}] = {{")
    for i in range(int(nr_class * (nr_class -1) / 2)):
        if(i < nr_class*(nr_class-1)/2 - 1):
            f.write(f"{model.rho[i]:.3f},")
        else:
            f.write(f"{model.rho[i]:.3f}")
    f.write("};\n\n")
    if(model.probA): # regression has probA only
        f.write("{")
        for i in range(int(nr_class*(nr_class-1)/2)):
            if(i < nr_class*(nr_class-1)/2 - 1):
                f.write(f"{model.probA[i]:.3f},")
            else:
                f.write(f"{model.probA[i]:.3f}")
            f.write("};\n\n")
    if(model.label):
        f.write(f"int label{suffix}[{nr_class}] = {{")
        for i in range(nr_class):
            if(i < nr_class - 1):
                f.write(f"{model.label[i]}")
            else:
                f.write(f"{model.label[i]}")
        f.write("};\n\n")
    if(model.nSV):
        f.write(f"int nSV{suffix}[{nr_class}] = {{")
        for i in range(nr_class):
            if(i < nr_class - 1):
                f.write(f"{model.nSV[i]},")
            else:
                f.write(f"{model.nSV[i]}")
        f.write("};\n\n")
    f.write(f"int free_sv{suffix} = 1;\n\n")

    f.write(f"struct svm_node *SV_ptr_arr{suffix}[{l}];\n\n")
    f.write(f"float *sv_coef_ptr_arr{suffix}[{nr_class - 1}];\n\n")

    if(param.svm_type <= 2):
        f.write(f"void init_SVC(struct svm_model* model{suffix})\n")
    else:
        f.write(f"void init_SVR(struct svm_model* model{suffix})\n")
    f.write("{\n")
    f.write(f"   model{suffix}->param = param{suffix};\n")
    f.write(f"   model{suffix}->l = l{suffix};\n")
    f.write(f"   model{suffix}->nr_class = nr_class{suffix};\n")
    f.write(f"   for(int i = 0; i < l{suffix}; i++){{\n")
    f.write(f"       SV_ptr_arr{suffix}[i] = SV{suffix}[i];\n")
    f.write(f"   }}\n")
    f.write(f"   sv_coef_ptr_arr{suffix}[0] = sv_coef{suffix}[0];\n")
    f.write(f"   model{suffix}->SV = SV_ptr_arr{suffix};\n")
    f.write(f"   model{suffix}->sv_coef = sv_coef_ptr_arr{suffix};\n")
    f.write(f"   model{suffix}->rho = rho{suffix};\n")
    if(model.nSV):
        f.write(f"   model{suffix}->nSV = nSV{suffix};\n")
    if(model.label):
        f.write(f"   model{suffix}->label = label{suffix};\n")
    f.write("}\n")

