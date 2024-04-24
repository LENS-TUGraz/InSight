float_precision = None

def add_init(pred_type, n_targets):
    if pred_type == "regression":
        return "float xgb_reg_predict(float* x)\n{\n\tfloat pred = 0;\n\n"
    elif pred_type == "classification":
        return "int xgb_class_predict(float* x)\n{\n\tfloat votes[" + str(n_targets) + "]= { 0.0 };\n\n"
    else:
        assert False

def add_split(tree, string, node, indents):
    string += "\t" * indents
    string += "if(x["
    string += tree[tree.Node == node].Feature.values[0][1:]
    string += "] < "
    if float_precision:
        string += str(round(tree[tree.Node == node].Split.values[0], float_precision))
    else:              
        string += str(tree[tree.Node == node].Split.values[0])
    string += ")\n"
    string += "\t" * indents
    string += "{\n"
    return string

def add_reg_leaf(tree, string, node, indents):
    string += "\t" * indents
    if float_precision:
        string += "pred += " + str(round(tree[tree.Node == node].Gain.values[0], float_precision)) + ";\n"
    else:              
        string += "pred += " + str(tree[tree.Node == node].Gain.values[0]) + ";\n"
    string += "\t" * (indents - 1) + "}\n"
    return string

def add_class_leaf(tree, string, node, indents, n_classes):
    class_label = tree.Tree.unique()[0] % n_classes
    string += "\t" * indents
    if float_precision:
        string += "votes[" + str(class_label) + "] += " + str(round(tree[tree.Node == node].Gain.values[0], float_precision)) + ";\n"
    else:              
        string += "votes[" + str(class_label) + "] += " + str(tree[tree.Node == node].Gain.values[0]) + ";\n"
    string += "\t" * (indents - 1) + "}\n"
    return string
    
def add_leaf(pred_type, tree, string, node, indents, n_classes):
    if pred_type == "regression":
        return add_reg_leaf(tree, string, node, indents)
    elif pred_type == "classification":
        return add_class_leaf(tree, string, node, indents, n_classes)
    else:
        assert False

def switch_branch(yes_branch, no_branch):
    return not yes_branch, not no_branch

def add_else_clause(string, indents):
    string += "\t" *  indents + "else\n"
    string += "\t" *  indents + "{\n"
    return string

def add_closed_bracket(string, indents):
    string += "\t" *  indents
    string += "}\n"
    return string

def add_reg_output(string, indents, xgb_tree):
    string += "\n"
    string += "\t" * indents
    string += f"""pred = {xgb_tree.get_params()["base_score"]} + pred;\n""";
    string += "\t" * indents
    string += "return pred;\n"
    string += "}"
    return string

def add_class_output(string, indents, n_classes):
    # string += "\n"
    # string += "\t" * indents
    # string += "uint8_t classIdx = 0;\n"
    # string += "\t" * indents
    # string += "float maxVotes = votes[0];\n"
    # string += "\t" * indents
    # string += "for (uint8_t i = 1; i < " + str(n_classes) + "; i++) {\n"
    # string += "\t" * (indents + 1)
    # string += "if (votes[i] > maxVotes) {\n"
    # string += "\t" * (indents + 2)
    # string += "classIdx = i;\n"
    # string += "\t" * (indents + 2)
    # string += "maxVotes = votes[i];\n"
    # string += "\t" * (indents + 1) + "}\n"
    # string += "\t" * indents + "}\n"
    # string += "\t" * indents
    # string += "return classIdx;\n"
    # string += "}"
    string += "\n"
    string += "\t" * indents
    string += "if(votes[0] > 0.5)\n"
    string += "\t" * (indents + 1)
    string += "return 1;\n"
    string += "\t" * indents
    string += "else\n"
    string += "\t" * (indents + 1)
    string += "return 0;\n"
    string += "}"
    return string

def add_output(pred_type, string, indents, xgb_tree, n_classes):
    if pred_type == "regression":
        return add_reg_output(string, indents, xgb_tree)
    elif pred_type == "classification":
        return add_class_output(string, indents, n_classes)
    else:
        assert False
        
def add_scale_values(string, scale_values, scale_type):
    string += "\n"
    string += "double scaling_" + scale_type + "[" + str(len(scale_values)) + "] = {"
    for idx, val in enumerate(scale_values):
        string += str(val)
        if idx < len(scale_values) - 1:
            string += ", "
    string += "};\n"
    return string

def is_leaf(tree, node):
    if tree[tree.Node == node].Split.isnull().values[0]:
        return True
    else:
        return False

def xgboost_converter(file_name, xgb, pred_type, n_classes, scale_min=[], scale_max=[]):
    string = add_init(pred_type, n_classes)
    df_xgb = xgb.get_booster().trees_to_dataframe()
    for idx in range(len(df_xgb.Tree.unique())):
        tree = df_xgb[df_xgb.Tree == idx]
        node = 0
        DONE = False
        yes_branch = True
        no_branch = False
        node_list = []
        function_indent = 1
        string += "\t" * function_indent + f"// tree #{idx + 1}\n"
        while not DONE:
            """ add split node """
            if not is_leaf(tree, node):
                string = add_split(tree, string, node, len(node_list) + function_indent)
                node_list.append(node)
                """ choose next node """
                if yes_branch:
                    node = int(tree[tree.Node == node].Yes.values[0].split("-")[-1])
                elif no_branch:
                    node = int(tree[tree.Node == node].No.values[0].split("-")[-1])
                    yes_branch, no_branch = switch_branch(yes_branch, no_branch)
        
            else: 
                """ add leaf node """
                string = add_leaf(pred_type, tree, string, node, len(node_list) + function_indent, n_classes)
                if len(node_list) == 0:
                    string = string[:-(len(node_list) + function_indent + 1)]
                    DONE = True
                    break
                """ choose next node """
                if yes_branch:
                    node = int(tree[tree.Node == node_list[-1]].No.values[0].split("-")[-1])
                    if is_leaf(tree, node):
                        yes_branch, no_branch = switch_branch(yes_branch, no_branch)
                    string = add_else_clause(string, len(node_list) + function_indent - 1)
                else: 
                    while node == int(tree[tree.Node == node_list[-1]].No.values[0].split("-")[-1]):
                        node = node_list.pop()
                        if len(node_list) == 0:
                            DONE = True
                            break
                        string = add_closed_bracket(string, len(node_list) + function_indent - 1)
                    if not DONE:
                        node = int(tree[tree.Node == node_list[-1]].No.values[0].split("-")[-1])
                        if not is_leaf(tree, node):
                            yes_branch, no_branch = switch_branch(yes_branch, no_branch)
                        string = add_else_clause(string, len(node_list) + function_indent - 1)
    string = add_output(pred_type, string, function_indent, xgb, n_classes)
    
    if len(scale_min) > 0:
        string = add_scale_values(string, scale_min, "min")
    if len(scale_max) > 0:
        string = add_scale_values(string, scale_max, "max")
            
    if pred_type == "regression":
        f = open(f"{file_name}", "w")
    elif pred_type == "classification":
        f = open(f"{file_name}", "w")
    else:
        assert False
    f.write(string)
    f.close()